
import argparse
import os
from submission.predictor import ImmuneStatePredictor

def main(train_dir: str, test_dir: str, out_dir: str, n_jobs: int = 4, device: str = "cpu"):
    predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device)
    predictor.fit(train_dir=train_dir, out_dir=out_dir)
    predictor.predict(train_dir=train_dir, test_dir=test_dir, out_dir=out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    main(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        device=args.device,
    )
