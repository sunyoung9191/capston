preprocessing 폴더:  눈썹이나 다른 요소가 들어가면 학습이 어려워지기 때문에, 모델 정확도를 위해 사진을 전부 눈만 crop되고 224x224로 resizing & padding되도록 만듦. 

model_similarity 폴더: 쌍꺼풀(class 3: 겉쌍, 속쌍(짝눈), 무쌍)으로 분류되어 만들어진 모델

model 폴더: randmark를 찍어서 눈 모양을 학습시키는 모델

