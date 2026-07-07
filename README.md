# 시맨틱 세그멘테이션 — FCN & U-Net / Semantic Segmentation

> **English summary.** Pixel-level image segmentation in PyTorch: a **Fully Convolutional Network (FCN)** that replaces dense layers with convolutions and upsampling to produce dense per-pixel predictions, and a **U-Net** for multi-class segmentation with its encoder–decoder + skip-connection architecture (image/mask pairs, FCN8s-style decoding).

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-CV-EE4C2C)

---

## 개요

이미지를 **픽셀 단위로 분류**하는 시맨틱 세그멘테이션을 두 대표 구조(FCN, U-Net)로 구현한 노트북입니다. 분류(class)·탐지(box)를 넘어, 객체의 **경계와 형태**를 픽셀 마스크로 예측합니다.

## 노트북 구성

| # | 노트북 | 내용 |
|---|--------|------|
| 01 | [fcn_segmentation](notebooks/01_fcn_segmentation.ipynb) | **FCN** — 사전학습 backbone + 완전합성곱·업샘플링으로 픽셀 예측, 학습 지표·가중치 저장/로드 |
| 02 | [unet_segmentation](notebooks/02_unet_segmentation.ipynb) | **U-Net** 멀티클래스 세그멘테이션 — 이미지·마스크 쌍 확인, 인코더-디코더 + skip-connection, FCN8s 디코딩 |

## 다룬 개념

- **FCN**: dense layer를 conv로 대체 → 임의 크기 입력에서 픽셀별 예측, 업샘플링(transpose conv)
- **U-Net**: 인코더-디코더 대칭 구조 + skip-connection으로 세밀한 경계 복원
- 데이터: 이미지–마스크 쌍 처리, 다중 클래스 마스크 시각화, 세그멘테이션 지표

## 실행 방법

```bash
pip install torch torchvision opencv-python pillow numpy pandas matplotlib scikit-learn tqdm
# 이미지/마스크 데이터셋을 노트북 경로에 배치 후 실행
jupyter notebook
```

## 참고

- 객체 탐지: [pytorch-object-detection](https://github.com/NvidiaSeoul/pytorch-object-detection)
- OpenCV 영상처리: [opencv-image-processing](https://github.com/NvidiaSeoul/opencv-image-processing)

---
> NVIDIA AI Academy Seoul 교육과정 실습 — [전체 포트폴리오](https://github.com/NvidiaSeoul)
