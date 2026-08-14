.PHONY: help train export-gguf export-gguf-random parity-gguf publish-hf test docker-build clean fuse-2026 validate-2026

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Training / export / publish workflow:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

train: ## Train BalancedStarlightDetector (long-running)
	python3 trainer.py --epochs 20 --batch_size 16 --out models/detector_balanced.pth

export-gguf: ## Export detector_balanced.pth to GGUF
	python3 scripts/export_starlight_gguf.py \
		--input models/detector_balanced.pth \
		--output models/starlight.gguf \
		--name-map models/starlight_gguf_map.json

export-gguf-random: ## Init-random GGUF export (CI / smoke)
	python3 scripts/export_starlight_gguf.py \
		--init-random \
		--output models/starlight_random.gguf \
		--name-map models/starlight_gguf_map.json

parity-gguf: ## PyTorch vs GGUF parity check
	python3 scripts/parity_starlight_gguf.py

publish-hf: ## Publish GGUF to Hugging Face (requires HF_TOKEN)
	./scripts/publish_to_hf.sh

test: ## Light sanity checks (imports + export --help)
	python3 -c "import trainer; assert hasattr(trainer, 'BalancedStarlightDetector'); print('trainer OK')"
	python3 scripts/export_starlight_gguf.py --help >/dev/null && echo 'export-gguf OK'
	@if command -v pytest >/dev/null 2>&1; then \
		python3 -m pytest tests/test_unified_pipeline.py -q --tb=no 2>/dev/null || true; \
	fi

docker-build: ## Build training/export image as starlight-train:latest
	docker build -t starlight-train:latest .

fuse-2026: ## Download SLM GGUF and bind official detector (grok_submission_2026)
	python3 datasets/grok_submission_2026/fuse_vlm.py

validate-2026: ## Run 2026 competition gates on the Grok reference entry
	python3 scripts/validate_submission_2026.py --submission datasets/grok_submission_2026

clean: ## Remove local smoke artifacts
	@rm -f models/starlight_random.gguf
	@rm -rf models_dist
	@echo "Clean complete"
