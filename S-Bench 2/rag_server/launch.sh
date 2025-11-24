#tmux new -s rag_server
#conda activate searchr1
#bash retrieval_launch.sh
#tmux detach
#bash train_ppo.sh

nvcc --version

#file_path=/your/path/to/data/rag_data
index_file=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/data/data/e5_Flat.index
corpus_file=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/data/data/wiki-18.jsonl
retriever=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/data/models/e5-base-v2

python3 rag_server/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever &
sleep 1200
