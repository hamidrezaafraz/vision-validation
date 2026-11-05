alex_reset_keep:
	uv run main.py cls --datadir="emotion/Data" --resultdir="emotion/rest" --verify_exact_duplicates --verify_near_duplicates  --hamming_threshold=4 --verify_near_embeddings  --restart --keep_first_in_group
facenet_vggface2:
	uv run main.py cls --datadir="emotion/Data" --resultdir="emotion/rest" --verify_near_embeddings --embedding_model_name="facenet-vggface2" --keep_first_in_group
facenet_casia_webface:
	uv run main.py cls --datadir="emotion/Data" --resultdir="emotion/rest" --verify_near_embeddings --embedding_model_name="facenet-casia-webface" --keep_first_in_group

alex_with_vggface2: alex_reset_keep facenet_vggface2, facenet_casia_webface

alex_with_casia_webface: alex_reset_keep facenet_casia_webface, facenet_vggface2