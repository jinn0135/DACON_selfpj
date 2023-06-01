# DACON 도배 하자 유형 분류 AI 경진대회

- https://dacon.io/competitions/official/236082/overview/description
- 팀 CNN2RNN (61위)

## 성능향상
- 모델 바꿈(ResNet -> EfficientNet)
- 하이퍼파라미터 조정(img resize, batch_size, learning_rate, ect)
- 주어진 데이터가 불균형했기 때문에 open cv를 이용해 적은 데이터를 증강시켜 준 후 학습
