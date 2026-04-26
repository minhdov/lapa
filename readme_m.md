pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install jax==0.4.23 jaxlib==0.4.23+cuda12.cudnn89 \
  jax-cuda12-pjrt==0.4.23 jax-cuda12-plugin==0.4.23 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install tensorflow==2.15.0 keras==2.15.0 tensorboard==2.15.2 \
  tensorflow-estimator==2.15.0 tensorflow-io-gcs-filesystem==0.37.1 ml-dtypes==0.2.0