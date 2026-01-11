.PHONY: help build test clean deploy

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: models_dist ## Build the Docker image
	docker build -t starlight-api:latest .

models_dist: ## Create models_dist directory with only PTH files
	@echo "Creating models_dist directory..."
	@mkdir -p models_dist
	@cp models/*.pth models_dist/ 2>/dev/null || true
	@cp models/*.json models_dist/ 2>/dev/null || true
	@cp models/*.md models_dist/ 2>/dev/null || true
	@echo "models_dist created with:"
	@ls -lh models_dist/

test: build ## Test the image
	@echo "Testing image..."
	docker run --rm starlight-api:latest python -c "import torch; print('✓ Dependencies OK')"
	docker run --rm starlight-api:latest ls -lh /app/models/

clean: ## Remove built images and models_dist
	@echo "Cleaning up..."
	@rm -rf models_dist
	@docker rmi starlight-api:latest 2>/dev/null || true
	@echo "Clean complete"

deploy: build ## Deploy image (already tagged as latest)
	@echo "Image tagged as starlight-api:latest"
	@echo "Ready to deploy with: docker push starlight-api:latest"

size: ## Show image sizes
	@echo "Docker image sizes:"
	@docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep starlight-api || true

all: clean build test ## Clean, build, and test
