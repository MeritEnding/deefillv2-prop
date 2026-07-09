// 인페인팅 스튜디오의 작업 모드. 제거 계열(mask+inpaint)은 프리셋만 다르고,
// outpaint는 캔버스를 확장하는 별도 흐름이다.
export const MODES = [
  {
    id: 'object',
    name: '객체 지우개',
    icon: 'eraser',
    short: '사람·물건 제거',
    hint: '지우고 싶은 사람이나 물건을 칠하세요.',
    brush: 34,
  },
  {
    id: 'watermark',
    name: '워터마크·자막 제거',
    icon: 'text',
    short: '워터마크·로고·자막',
    hint: '워터마크, 로고, 자막 위를 얇게 칠하세요.',
    brush: 22,
  },
  {
    id: 'restore',
    name: '사진 복원',
    icon: 'sparkle',
    short: '스크래치·손상 복구',
    hint: '스크래치, 접힌 자국, 얼룩진 부분을 칠하세요.',
    brush: 16,
  },
  {
    id: 'anonymize',
    name: '얼굴·번호판 익명화',
    icon: 'shield',
    short: '자연스러운 익명화',
    hint: '가리고 싶은 얼굴이나 번호판을 칠하세요.',
    brush: 30,
  },
  {
    id: 'declutter',
    name: '홈스테이징',
    icon: 'home',
    short: '가구·잡동사니 비우기',
    hint: '비우고 싶은 가구나 물건을 칠하세요.',
    brush: 42,
  },
  {
    id: 'batch',
    name: '일괄 지우기',
    icon: 'layers',
    short: '여러 장 한 번에',
    hint: '같은 위치의 워터마크를 여러 장에서 한 번에 지웁니다.',
    brush: 26,
    special: 'batch',
  },
  {
    id: 'outpaint',
    name: '배경 확장',
    icon: 'expand',
    short: '아웃페인팅',
    hint: '확장할 방향과 비율을 고르면 바깥 배경을 채웁니다.',
    special: 'outpaint',
  },
]

export const MODE_MAP = Object.fromEntries(MODES.map((m) => [m.id, m]))
