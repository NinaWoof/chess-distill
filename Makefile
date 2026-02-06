.PHONY: setup gen train play test lint

setup:
	uv pip install -e .

gen:
	python scripts/gen_labels.py --max_positions 1000

train:
	python scripts/train_policy_value.py --epochs 2

play:
	python scripts/play_cli.py --ckpt checkpoints/latest.pt

test:
	pytest tests/

lint:
	ruff check .
	ruff format .
