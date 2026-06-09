ARG COGNAV_BASE_IMAGE=cognav-vln:1.0
FROM ${COGNAV_BASE_IMAGE}

ARG CONDA_ENV_NAME=cognav
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:/opt/conda/bin:${PATH}
ENV GROUNDING_DINO_PATH=/opt/mmdetection

SHELL ["/bin/bash", "-lc"]

RUN pip install --no-cache-dir "mmengine==0.10.7" \
 && pip install --no-cache-dir \
      "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv-2.1.0-cp39-cp39-manylinux1_x86_64.whl" \
 && pip install --no-cache-dir "mmdet==3.3.0" "segment-anything==1.0" \
 && pip install --no-cache-dir "pathfinding==1.0.9" "bresenham==0.2.1" "supervision==0.26.1" \
 && python -c "from pathlib import Path; import mmdet; pkg=Path(mmdet.__file__).resolve().parent; configs=pkg/'.mim'/'configs'; target=Path('/opt/mmdetection'); target.mkdir(parents=True, exist_ok=True); (target/'configs').symlink_to(configs); print('mmdet configs:', configs)" \
 && pip install --no-cache-dir "requests==2.32.5" "urllib3==2.6.3" "setuptools==64.0.3" \
 && python -c "import mmdet; from mmdet.apis import DetInferencer; from segment_anything import build_sam; print('mmdet', mmdet.__version__); print('STRIVE detector imports OK')"

WORKDIR /workspace/STRIVE
