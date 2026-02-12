sudo chmod -R +x /mnt/shared-storage-user/ailab-llmkernel/yangkaichen/Code_judgement/reward_server/launch_reward_api.sh
rjob submit --name=ykc-reward-api \
--gpu=8 \
--memory=512000 \
--cpu=64 \
--charged-group=llmkernel_gpu \
--private-machine=group \
--priority=6 \
--mount=gpfs://gpfs1/ailab-llmkernel:/mnt/shared-storage-user/ailab-llmkernel \
--mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
--image=registry.h.pjlab.org.cn/ailab-llmkernel-llmkernel_gpu/gongjingyang-workspace:0908 \
--host-network=true \
-P 1 \
-- bash /mnt/shared-storage-user/ailab-llmkernel/yangkaichen/Code_judgement/reward_server/launch_reward_api.sh