# Handplay Music

손 동작과 손 모양으로 친구들과 함께 음악을 다루는 웹 실험입니다.

## Run

```sh
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Audio

기본 버튼은 아래 파일명을 찾습니다.

- `audio/track-01.mp3`
- `audio/track-02.mp3`
- `audio/track-03.mp3`

파일명을 바꾸고 싶으면 `app.js`의 `BUILT_IN_TRACKS`를 수정하면 됩니다. 화면의 `MP3` 버튼으로 로컬 음원을 바로 불러올 수도 있습니다.

## Gesture

- 볼륨: 손을 웅크리면 작아지고, 손을 펼수록 커짐. 완전히 펼친 손은 볼륨 100
- 트랙 변경: 펼친 손가락 개수로 음원을 넣은 순서의 트랙을 선택
- 손 모양: 오므리면 응집된 소리, 펼치면 퍼지는 소리
