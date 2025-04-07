export CONTAINER_NAME=terry
export NONE_DOCKER_IMAGES=`docker images -f dangling=true -q`
#================================================
b: ## build docker image
# docker compose -f docker-compose.gpu.yml build --no-cache # DO WE NEED NO-CACHE?
	docker compose -f docker-compose.gpu.yml build
uc: ## docker-compose up -d and connect the container
	@make u
	@make c
u: ## docker-compose up -d
	docker compose -f docker-compose.gpu.yml up -d
c: ## connect newest container
	docker exec -it $(CONTAINER_NAME) bash
rmi-none: ## remove NONE images
	docker rmi $(NONE_DOCKER_IMAGES) -f
#================================================
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
