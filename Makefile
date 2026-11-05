alex_reset_keep:
	uv run main.py cls --datadir="emotion/Data" --resultdir="emotion/rest" \
	--verify_exact_duplicates --verify_near_duplicates  --hamming_threshold=4 \
	--verify_near_embeddings  \
	--domain_expert_embedding --domain_expert_model_names facenet-vggface2 facenet-casia-webface \
	--restart --keep_first_in_group