# Planet Escape

## 프로젝트 개요

### **Physics based Space Golf Game**

|<img src="https://github.com/user-attachments/assets/7e11c802-8f3d-40f2-82be-96cf12f1ecd8" alt="Ingame_screen" width="400" height="300" style="margin:0; padding:0;">| <img src="https://github.com/user-attachments/assets/2287a3e4-e0a5-4a25-b14a-ee791431b76c" alt="Gravitymap" width="400" height="300" style="margin:0; padding:0;">|
|:-----------------:|:----------------:|
|Ingame screen|Gravity Map|

- **Planet Escape**에서 플레이어는 조난당한 우주선의 **선장**입니다. 연료는 한 번의 추진만 가능한 정도만 남아서 **단 한 번의 발사**로 **우주선을 탈출**시켜야합니다. 
- **중력지도**를 통해 행성의 중력을 계산하여 행성과 소행성대의 **충돌을 피해** 목표지점으로 안전히 탈출하세요.
### **Physics**

- 오브젝트 사이의 충돌 처리
- 질량에 따른 중력을 실현한 물리 엔진 구현
- matplot을 통한 gravity map 시각화

## 조작 방법&구조
### 조작방법
- 플레이어(파란색)은 좌우 키로 발사각도를 상하키를 발사의 세기를 조절할수있습니다.
- shift 키를 누를 시 더세밀한 조작이 가능합니다.
- space 키를 통해 우주선을 발사할 수있습니다.
- R 키를 통해 레벨을 다시 시작가능합니다.

- M 키를 통해 중력지도를 볼 수 있습니다.
### 오브젝트
- 행성은 고유의 질량을 가지고 있어 플레이어를 끌어당기고 플레이어는 이 행성에 충돌하면 안됩니다. 여기서 주의 할 점은 행성의 크기는 질량에 비례하지 않는다는 점입니다. 어떤행성은 가스로 이루어져 있어 크기에 비해 가벼울 수 있고 어떤 행성은 작지만 무거운 금속으로 이루어져 플레이어를 강하게 끌어당길 수 있습니다. 
- 소행성대는 플레이어를 끌어당기지는 않지만 무수한 소행성이 띠의 형태를 이루고 있어 플레이어가 지나가는 것을 막습니다. 물론 소행성과 충돌하지 않고 띠를 통과하는 방법도 있겠지만 모든 중력을 계산하고 띠를 통과하는 것은 불가능한 일에 가까울 것입니다.
- 밝은 하늘색을 띄는 초신성은 강력한 폭발로 인한 가스분출로 큰 빛을 내며 플레이어를 밀어냅니다. 플레이어는 밀어내는 정도를 중력지도에서 확인가능하고 이를 통해 종래에는 불가능하던 새로운 경로로의 이동이 가능합니다. 물론, 행성과 마찬가지로 폭발에 너무 접근하면 플레이어는 타버리게 됩니다.

## 특징 설명

### **오브젝트 사이의 충돌 처리**

> 충돌 여부의 계산을 최적화하기위해 collision 영역은 원으로 설정하여 두 원사이의 중심의 거리와 반지를의 합을 비교하는 방식으로 최적화 하였습니다.

> 소행성대의 경우 플레이어와 소행성대의 모든 소행성 오브젝트 사이의 충돌을 바로 계산하지 않고 소행성대의 collision영역을 도넛의 형태로 따로 두어 소행성대와 플레이어가 충돌한경우만 내부 소행성과 충돌여부를 계산하는 방식으로 최적화 하였습니다.

### **물리 엔진 구현**
> 질량과 거리를 바탕으로 계산한 중력은 플레이어 에게 행성 방향으로 지속적으로 가속도를 부여합니다.

## 데모 영상

https://www.youtube.com/watch?v=jUoGzTEvR8w

## 실행 파일

- Python 3.11.4
