import os
import re
import json
import argparse
from typing import Literal

_DatasetType = Literal['train', 'test']

def get_last_n_ckpt_names(ckpt_dir: str, n: int):
    """Get the last n checkpoints in the directory."""
    if not os.path.exists(ckpt_dir):
        return None

    ckpts = [f for f in os.listdir(ckpt_dir) if re.search(r'epoch_(\d+).pth$', f)]
    if len(ckpts) == 0:
        return None

    ckpts.sort()
    return ckpts[-n:]

def get_last_n_ckpt_eval_results(ckpt_dir: str, n: int, dataset_type: _DatasetType):
    eval_result_name_template = os.path.join(ckpt_dir, 'eval_{ckpt_name_base}_{dataset_type}_cameras', 'success_by_seed.json')
    ckpt_names = get_last_n_ckpt_names(ckpt_dir, n)

    eval_results = []
    for ckpt_name in ckpt_names:
        ckpt_name_base = ckpt_name.replace('.pth', '')
        eval_results.append(eval_result_name_template.format(ckpt_name_base=ckpt_name_base))

    return eval_results

def calc_success_rate(ckpt_dir: str, n: int = 10, dataset_type: _DatasetType = 'test'):
    eval_results = get_last_n_ckpt_eval_results(ckpt_dir, n, dataset_type)

    results = {}
    for eval_result in eval_results:
        if not os.path.exists(eval_result):
            continue

        with open(eval_result, 'r') as f:
            data = json.load(f)

        successes = list(data.values())
        num_true = sum(successes)
        total = len(successes)
        success_rate = num_true / total if total > 0 else 0

        # Extract epoch from path
        match = re.search(r'epoch_(\d+)', eval_result)
        epoch = int(match.group(1)) if match else -1

        results[epoch] = {
            'num_true': num_true,
            'total': total,
            'success_rate': success_rate
        }

    # Sort by epoch
    sorted_epochs = sorted(results.keys())
    for epoch in sorted_epochs:
        res = results[epoch]
        print(f"Epoch {epoch}: {res['num_true']}/{res['total']} success rate: {res['success_rate']:.2%}")

    if len(results) > 0:
        avg_success_rate = sum(res['success_rate'] for res in results.values()) / len(results)
        print(f"\nAverage Success Rate across {len(results)} checkpoints: {avg_success_rate:.2%}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/root/Desktop/workspace/CamPoseOpensource-VAI/policy_robosuite/checkpoints/train_dp_use_plucker_liftrand_eef_delta",
        help="Path to the checkpoint directory"
    )
    parser.add_argument("--n", type=int, default=10, help="Number of checkpoints to evaluate")
    parser.add_argument("--dataset", type=str, default="test", choices=["train", "test"])

    args = parser.parse_args()

    calc_success_rate(args.ckpt_dir, args.n, args.dataset)
