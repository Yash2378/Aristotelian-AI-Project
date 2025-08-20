.PHONY: help setup-dev run-dev test lint format clean build docker-build docker-run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@egrep '^(.+)\s*:.*?##\s*(.+)$' $(MAKEFILE_LIST) | column -t -c 2 -s ':#'

setup-dev: ## Setup development environment
	python -m pip install --upgrade pip
	pip install -r requirements/development.txt
	pre-commit install
	python scripts/setup/download_models.py

init-db: ## Initialize database
	python scripts/setup/initialize_database.py

download-models: ## Download required language models
	python scripts/setup/download_models.py

run-dev: ## Run development server
	python -m src.api.main

test: ## Run tests
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

build: ## Build package
	python -m build

docker-build: ## Build Docker image
	docker build -t aristotelian-ai:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env aristotelian-ai:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting conversation")

async def run_real_time_evaluation(
    orchestrator: ComprehensiveEvaluationOrchestrator,
    request: ChatRequest,
    response: ChatResponse
):
    """Background task for real-time evaluation"""
    try:
        # Prepare evaluation data
        test_data = {
            "responses": {response.language: response.response},
            "expected_concepts": response.philosophical_concepts,
            "cultural_context": {response.language: request.cultural_context or {}},
            "interaction_data": [{
                "user_id": request.user_id,
                "message": request.message,
                "response": response.response,
                "language": response.language,
                "timestamp": "now",
                "processing_time": response.processing_time
            }]
        }
        
        # Run evaluation
        evaluation_results = await orchestrator.run_comprehensive_evaluation(test_data)
        
        # Store results for analytics
        await orchestrator.store_real_time_evaluation(
            user_id=request.user_id,
            conversation_id=response.conversation_id,
            evaluation_results=evaluation_results
        )
        
    except Exception as e:
        logger.error(f"Error in real-time evaluation: {e}")