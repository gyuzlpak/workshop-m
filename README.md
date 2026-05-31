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

- 볼륨: 한 손은 검지만 펴서 볼륨 50의 기준점을 만들고, 반대 손 검지를 그 기준점보다 위/아래로 움직여 조절
- 손 모양: 오므리면 응집된 소리, 펼치면 퍼지는 소리
