import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from src.agents import CPPOAgent, PPOConfig
from src.envs import GraphDefenseEnv
from src.models import GAT, GCN
from src.utils import compute_graph_stats
from src.utils.early_stopping import EarlyStopping


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)


def setup_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logging.info("Using device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_{args.model_type}_{timestamp}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    return device, output_dir


def load_clean_dataset(dataset_name: str, data_dir: str = "data") -> Data:
    if dataset_name not in ["Cora", "CiteSeer", "PubMed"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = Planetoid(root=os.path.join(data_dir, dataset_name), name=dataset_name)
    data = dataset[0]

    stats = compute_graph_stats(data)
    logging.info("Clean dataset %s loaded", dataset_name)
    for key, value in stats.items():
        logging.info("  %s: %s", key, value)

    return data


def _to_bool_mask(mask: torch.Tensor, n: int) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype in (torch.int32, torch.int64):
        if mask.numel() == n:
            return mask.bool()
        out = torch.zeros(n, dtype=torch.bool)
        out[mask.long()] = True
        return out
    return mask.bool()


def load_poisoned_data(path: str) -> Tuple[Data, Optional[torch.Tensor], Optional[int]]:
    """
    Expected formats:
    1) torch_geometric.data.Data
    2) dict with key 'data' or 'poisoned_data' + optional 'attack_mask'/'poison_mask' + optional 'target_label'
    """
    obj = torch.load(path, map_location="cpu")

    poisoned_data: Optional[Data] = None
    attack_mask: Optional[torch.Tensor] = None
    target_label: Optional[int] = None

    if isinstance(obj, Data):
        poisoned_data = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("data", None), Data):
            poisoned_data = obj["data"]
        elif isinstance(obj.get("poisoned_data", None), Data):
            poisoned_data = obj["poisoned_data"]

        if "attack_mask" in obj and obj["attack_mask"] is not None:
            attack_mask = obj["attack_mask"]
        elif "poison_mask" in obj and obj["poison_mask"] is not None:
            attack_mask = obj["poison_mask"]

        if "target_label" in obj and obj["target_label"] is not None:
            target_label = int(obj["target_label"])
    else:
        raise ValueError(f"Unsupported poisoned data file format: {type(obj)}")

    if poisoned_data is None:
        raise ValueError("No Data object found in poisoned data file")

    n = poisoned_data.num_nodes
    if attack_mask is not None:
        attack_mask = _to_bool_mask(attack_mask, n)

    if target_label is None and hasattr(poisoned_data, "target_label"):
        try:
            target_label = int(poisoned_data.target_label)
        except Exception:
            target_label = None

    logging.info("Poisoned data loaded from %s", path)
    logging.info("  num_nodes=%d, num_edges=%d", poisoned_data.num_nodes, poisoned_data.edge_index.size(1))
    if attack_mask is not None:
        logging.info("  attack_mask_count=%d", int(attack_mask.sum().item()))
    if target_label is not None:
        logging.info("  target_label=%d", target_label)

    return poisoned_data, attack_mask, target_label


def train_base_model(
    data: Data,
    model_type: str = "GCN",
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    device: str = "cpu",
    es_patience: int = 50,
    es_min_delta: float = 0.001,
) -> torch.nn.Module:
    logging.info("Training base %s model", model_type)

    if model_type == "GCN":
        model = GCN(in_dim=data.x.size(1), hid=64, out_dim=int(data.y.max().item() + 1), dropout=0.5).to(device)
    elif model_type == "GAT":
        model = GAT(in_dim=data.x.size(1), hid=64, out_dim=int(data.y.max().item() + 1), dropout=0.5).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    data = data.to(device)

    best_val_acc = 0.0
    val_acc = 0.0
    early_stopping = EarlyStopping(
        patience=es_patience,
        min_delta=es_min_delta,
        mode="max",
        verbose=True,
        save_path=None,
    )

    pbar = tqdm(range(epochs), desc="Base Model", unit="epoch")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                pred = logits.argmax(dim=-1)
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
                best_val_acc = max(best_val_acc, val_acc)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "train": f"{train_acc:.4f}",
                        "val": f"{val_acc:.4f}",
                        "test": f"{test_acc:.4f}",
                    }
                )

        if early_stopping(val_acc, model=None):
            logging.info("Early stop base model at epoch %d", epoch + 1)
            break

    pbar.close()
    logging.info("Base model done. best_val_acc=%.4f", best_val_acc)
    return model


@torch.no_grad()
def evaluate_defense(
    model,
    clean_data: Data,
    poisoned_data: Data,
    device: str = "cpu",
    target_label: Optional[int] = 0,
    attack_mask: Optional[torch.Tensor] = None,
    edge_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()
    clean_data = clean_data.to(device)
    poisoned_data = poisoned_data.to(device)

    clean_logits = model(clean_data.x, clean_data.edge_index)
    clean_pred = clean_logits.argmax(dim=-1)
    clean_mask = clean_data.test_mask if hasattr(clean_data, "test_mask") else torch.ones(clean_data.num_nodes, dtype=torch.bool, device=device)
    clean_acc = (clean_pred[clean_mask] == clean_data.y[clean_mask]).float().mean().item()

    poison_logits = model(poisoned_data.x, poisoned_data.edge_index, edge_weight=edge_weight)
    poison_pred = poison_logits.argmax(dim=-1)

    if attack_mask is None:
        if hasattr(poisoned_data, "attack_mask"):
            eval_mask = poisoned_data.attack_mask.bool()
        elif hasattr(poisoned_data, "poison_mask"):
            eval_mask = poisoned_data.poison_mask.bool()
        elif hasattr(poisoned_data, "test_mask"):
            eval_mask = poisoned_data.test_mask
        else:
            eval_mask = torch.ones(poisoned_data.num_nodes, dtype=torch.bool, device=device)
    else:
        eval_mask = attack_mask.to(device).bool()

    if eval_mask.sum().item() == 0:
        asr = 0.0
    elif target_label is not None:
        asr = (poison_pred[eval_mask] == int(target_label)).float().mean().item()
    else:
        asr = (poison_pred[eval_mask] != poisoned_data.y[eval_mask]).float().mean().item()

    benign_mask = ~eval_mask
    if benign_mask.sum().item() > 0:
        benign_acc = (poison_pred[benign_mask] == poisoned_data.y[benign_mask]).float().mean().item()
    else:
        benign_acc = 0.0

    return {
        "clean_acc": clean_acc,
        "asr": asr,
        "benign_acc_poison_graph": benign_acc,
        "attack_eval_count": float(eval_mask.sum().item()),
    }


def run_al_rl_training(
    clean_data: Data,
    poisoned_data: Data,
    model,
    device: str,
    output_dir: str,
    args,
    attack_mask: Optional[torch.Tensor],
    target_label: Optional[int],
) -> Dict[str, List]:
    logging.info("Starting AL + CPPO defense training")

    tensorboard_dir = os.path.join(output_dir, "runs")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    env = GraphDefenseEnv(
        poisoned_data=poisoned_data,
        model=model,
        device=device,
        topk=args.topk,
        lambda_asr=args.lambda_asr,
        lambda_acc=args.lambda_acc,
        lambda_cost=args.lambda_cost,
        max_steps=args.max_steps_per_episode,
        target_label=target_label,
        poisoned_eval_mask=attack_mask,
        asr_eval_frequency=args.asr_eval_frequency,
        min_clean_acc=args.min_clean_acc,
        clean_acc_penalty=args.clean_acc_penalty,
        anomaly_weight=args.anomaly_weight,
        jaccard_threshold=args.jaccard_threshold,
        guard_beta=args.guard_beta,
        max_local_prune_edges=args.max_local_prune_edges,
    )

    ppo_config = PPOConfig(
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        lr=args.lr_rl,
        train_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        dual_lr=args.dual_lr,
        budget_c=args.budget_c,
    )
    agent = CPPOAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size(),
        config=ppo_config,
        device=device,
    )

    history = {
        "episode_returns": [],
        "episode_costs": [],
        "clean_accs": [],
        "asrs": [],
        "lambda_values": [],
        "policy_losses": [],
        "value_losses": [],
    }

    rl_early_stopping = EarlyStopping(
        patience=args.rl_es_patience,
        min_delta=args.rl_es_min_delta,
        mode="min",
        verbose=True,
        save_path=os.path.join(output_dir, "best_rl_model.pt"),
    )

    ep_pbar = tqdm(range(args.num_episodes), desc="RL Train", unit="episode")
    for episode in ep_pbar:
        state = env.reset(budget_ratio=args.budget_ratio)

        traj = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "costs": [],
            "values": [],
            "safety_values": [],
            "dones": [],
        }

        ep_return = 0.0
        ep_cost = 0.0

        for _ in range(args.max_steps_per_episode):
            action, logprob, value, safety_value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            traj["states"].append(state)
            traj["actions"].append(action)
            traj["logprobs"].append(logprob)
            traj["rewards"].append(float(reward))
            traj["costs"].append(float(info.get("cost", 0.0)))
            traj["values"].append(value)
            traj["safety_values"].append(safety_value)
            traj["dones"].append(bool(done))

            ep_return += float(reward)
            ep_cost += float(info.get("cost", 0.0))
            state = next_state

            if done:
                break

        avg_step_cost = ep_cost / max(1, len(traj["states"]))
        update_info = agent.update([traj], avg_step_cost) if len(traj["states"]) > 0 else {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "lambda": 0.0,
        }

        history["episode_returns"].append(ep_return)
        history["episode_costs"].append(ep_cost)
        history["lambda_values"].append(update_info.get("lambda", 0.0))
        history["policy_losses"].append(update_info.get("policy_loss", 0.0))
        history["value_losses"].append(update_info.get("value_loss", 0.0))

        writer.add_scalar("RL/Episode_Return", ep_return, episode)
        writer.add_scalar("RL/Episode_Cost", ep_cost, episode)
        writer.add_scalar("RL/Lambda", update_info.get("lambda", 0.0), episode)
        writer.add_scalar("RL/Policy_Loss", update_info.get("policy_loss", 0.0), episode)
        writer.add_scalar("RL/Value_Loss", update_info.get("value_loss", 0.0), episode)

        if (episode + 1) % args.eval_frequency == 0:
            eval_results = evaluate_defense(
                model=model,
                clean_data=clean_data,
                poisoned_data=env.current_data,
                device=device,
                target_label=target_label,
                attack_mask=attack_mask,
                edge_weight=env.edge_weights,
            )
            history["clean_accs"].append(eval_results["clean_acc"])
            history["asrs"].append(eval_results["asr"])

            writer.add_scalar("Defense/Clean_Accuracy", eval_results["clean_acc"], episode)
            writer.add_scalar("Defense/ASR", eval_results["asr"], episode)
            writer.add_scalar("Defense/Benign_Acc_PoisonGraph", eval_results["benign_acc_poison_graph"], episode)

            ep_pbar.set_postfix(
                {
                    "Return": f"{ep_return:.3f}",
                    "Cost": f"{ep_cost:.2f}",
                    "Acc": f"{eval_results['clean_acc']:.3f}",
                    "ASR": f"{eval_results['asr']:.3f}",
                    "λ": f"{update_info.get('lambda', 0.0):.3f}",
                }
            )

            if rl_early_stopping(
                eval_results["asr"],
                agent.actor_critic,
                extra_info={"episode": episode + 1, "clean_acc": eval_results["clean_acc"]},
            ):
                logging.info("Early stop RL at episode %d", episode + 1)
                break
        else:
            ep_pbar.set_postfix(
                {
                    "Return": f"{ep_return:.3f}",
                    "Cost": f"{ep_cost:.2f}",
                    "λ": f"{update_info.get('lambda', 0.0):.3f}",
                }
            )

        if (episode + 1) % args.save_frequency == 0:
            ckpt = os.path.join(output_dir, f"checkpoint_episode_{episode+1}.pt")
            agent.save(ckpt)
            with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

    ep_pbar.close()

    if rl_early_stopping.early_stop and rl_early_stopping.save_path is not None:
        logging.info("Loading best RL model due to early stopping")
        agent.load(rl_early_stopping.save_path)

    final_results = evaluate_defense(
        model=model,
        clean_data=clean_data,
        poisoned_data=env.current_data if env.current_data is not None else poisoned_data,
        device=device,
        target_label=target_label,
        attack_mask=attack_mask,
        edge_weight=env.edge_weights,
    )

    agent.save(os.path.join(output_dir, "final_model.pt"))
    torch.save(
        {
            "defended_data": env.current_data.cpu() if env.current_data is not None else poisoned_data.cpu(),
            "edge_weights": env.edge_weights.cpu() if env.edge_weights is not None else None,
            "target_label": target_label,
            "attack_mask": attack_mask.cpu() if attack_mask is not None else None,
        },
        os.path.join(output_dir, "defended_graph.pt"),
    )
    with open(os.path.join(output_dir, "final_results.json"), "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    writer.close()
    logging.info("Final results: clean_acc=%.4f, asr=%.4f", final_results["clean_acc"], final_results["asr"])
    return history


def main():
    parser = argparse.ArgumentParser(description="A2GBD AL+CPPO defense (poison data provided)")

    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--model_type", type=str, default="GCN", choices=["GCN", "GAT"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--poisoned_data_path", type=str, required=True, help="Path to poisoned graph .pt file")
    parser.add_argument("--target_label", type=int, default=None, help="Override target label for ASR")

    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps_per_episode", type=int, default=32)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--budget_ratio", type=float, default=0.005)

    parser.add_argument("--lambda_asr", type=float, default=1.0)
    parser.add_argument("--lambda_acc", type=float, default=0.5)
    parser.add_argument("--lambda_cost", type=float, default=0.1)

    parser.add_argument("--lr_rl", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--dual_lr", type=float, default=1e-2)
    parser.add_argument("--budget_c", type=float, default=10.0)

    parser.add_argument("--base_epochs", type=int, default=200)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--base_model_path", type=str, default="", help="Optional pre-trained base model state_dict path")

    parser.add_argument("--base_es_patience", type=int, default=50)
    parser.add_argument("--base_es_min_delta", type=float, default=0.001)
    parser.add_argument("--rl_es_patience", type=int, default=20)
    parser.add_argument("--rl_es_min_delta", type=float, default=0.005)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--eval_frequency", type=int, default=10)
    parser.add_argument("--save_frequency", type=int, default=50)
    parser.add_argument("--asr_eval_frequency", type=int, default=1)
    parser.add_argument("--min_clean_acc", type=float, default=0.0, help="Soft clean-accuracy floor in RL reward")
    parser.add_argument("--clean_acc_penalty", type=float, default=0.0, help="Penalty weight when clean_acc < min_clean_acc")
    parser.add_argument("--anomaly_weight", type=float, default=0.7, help="Weight of anomaly score in candidate ranking")
    parser.add_argument("--jaccard_threshold", type=float, default=0.05, help="Threshold for local Jaccard edge pruning")
    parser.add_argument("--guard_beta", type=float, default=3.0, help="Sharpness for guard-style edge reweighting")
    parser.add_argument("--max_local_prune_edges", type=int, default=2, help="Max edges removed per local prune action")

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    device, output_dir = setup_experiment(args)

    clean_data = load_clean_dataset(args.dataset, args.data_dir)
    poisoned_data, attack_mask, poison_target_label = load_poisoned_data(args.poisoned_data_path)

    target_label = args.target_label if args.target_label is not None else poison_target_label

    if args.base_model_path:
        if args.model_type == "GCN":
            model = GCN(
                in_dim=clean_data.x.size(1),
                hid=64,
                out_dim=int(clean_data.y.max().item() + 1),
                dropout=0.5,
            ).to(device)
        else:
            model = GAT(
                in_dim=clean_data.x.size(1),
                hid=64,
                out_dim=int(clean_data.y.max().item() + 1),
                dropout=0.5,
            ).to(device)
        model.load_state_dict(torch.load(args.base_model_path, map_location=device))
        logging.info("Loaded base model from %s", args.base_model_path)
    else:
        model = train_base_model(
            clean_data,
            model_type=args.model_type,
            epochs=args.base_epochs,
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            device=device,
            es_patience=args.base_es_patience,
            es_min_delta=args.base_es_min_delta,
        )

    torch.save(model.state_dict(), os.path.join(output_dir, "base_model.pt"))

    run_al_rl_training(
        clean_data=clean_data,
        poisoned_data=poisoned_data,
        model=model,
        device=device,
        output_dir=output_dir,
        args=args,
        attack_mask=attack_mask,
        target_label=target_label,
    )

    logging.info("Training completed. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
