alex_reset_keep:
	uv run main.py cls --datadir="emotion/Data" --resultdir="emotion/rest" \
	--verify_exact_duplicates --verify_near_duplicates  --hamming_threshold=4 \
	--verify_near_embeddings  \
	--domain_expert_embedding --domain_expert_model_names facenet-vggface2 facenet-casia-webface \
	--restart --keep_first_in_group

data:
	uv run main.py data --datadir="emotion/Data" --resultdir="emotion/results" \
	--group_hash \
	--group_perceptual_hash --perceptual_hash phash --hamming_threshold=4 \
	--group_embeddings --embedding_model_name alexnet --embedding_threshold 0.95 

data_reset:
	uv run main.py data --datadir="emotion/Data" --resultdir="emotion/results" \
	--group_hash \
	--group_perceptual_hash --perceptual_hash phash --hamming_threshold=4 \
	--restart