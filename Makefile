docker.up:
	docker network create langchain || true; \
	docker compose -f ./docker-compose.yml up -d