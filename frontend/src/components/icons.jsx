// 얇은 라인 아이콘 세트 (currentColor 사용).
const S = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round' }

export const Icon = {
  eraser: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M7.5 18.5 4 15a2 2 0 0 1 0-2.8l7-7a2 2 0 0 1 2.8 0l4.2 4.2a2 2 0 0 1 0 2.8l-6.7 6.7H8.5a1 1 0 0 1-.7-.3Z" />
      <path d="m9 11 5 5M4 21h16" />
    </svg>
  ),
  text: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M4 7V5h16v2M9 19h6M12 5v14" />
    </svg>
  ),
  sparkle: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M12 3l1.8 4.9L18.7 9.7l-4.9 1.8L12 16.4l-1.8-4.9L5.3 9.7l4.9-1.8L12 3Z" />
      <path d="M19 15l.7 1.9 1.9.7-1.9.7-.7 1.9-.7-1.9-1.9-.7 1.9-.7Z" />
    </svg>
  ),
  shield: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M12 3l7 3v6c0 4.4-3 7.5-7 9-4-1.5-7-4.6-7-9V6l7-3Z" />
      <circle cx="12" cy="10.5" r="2" />
      <path d="M8.5 16c.6-1.6 2-2.5 3.5-2.5s2.9.9 3.5 2.5" />
    </svg>
  ),
  home: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M4 11l8-6 8 6M6 10v9h12v-9" />
      <path d="M10 19v-5h4v5" />
    </svg>
  ),
  expand: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3" />
      <rect x="8.5" y="8.5" width="7" height="7" rx="1" />
    </svg>
  ),
  bag: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M6 8h12l-1 12H7L6 8Z" />
      <path d="M9 8V6a3 3 0 0 1 6 0v2" />
    </svg>
  ),
  plane: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M10.5 13.5 3 11l2-2 6 1 4-4.5A2 2 0 0 1 18 6l-4.5 4 1 6-2 2-2.5-7.5-3 3v3l-1.5-1.5L4 17l3-3Z" />
    </svg>
  ),
  box: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3Z" />
      <path d="M4 7.5 12 12l8-4.5M12 12v9" />
    </svg>
  ),
  building: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <rect x="6" y="3" width="12" height="18" rx="1" />
      <path d="M9.5 7h1M13.5 7h1M9.5 11h1M13.5 11h1M9.5 15h1M13.5 15h1M10 21v-3h4v3" />
    </svg>
  ),
  bolt: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" />
    </svg>
  ),
  brush: (p) => (
    <svg viewBox="0 0 24 24" width="20" height="20" {...S} {...p}>
      <path d="M15 4 20 9 9 20l-5 1 1-5L15 4Z" />
    </svg>
  ),
  undo: (p) => (
    <svg viewBox="0 0 24 24" width="18" height="18" {...S} {...p}>
      <path d="M9 7 4 12l5 5M4 12h11a5 5 0 0 1 0 10h-1" />
    </svg>
  ),
  redo: (p) => (
    <svg viewBox="0 0 24 24" width="18" height="18" {...S} {...p}>
      <path d="m15 7 5 5-5 5M20 12H9a5 5 0 0 0 0 10h1" />
    </svg>
  ),
  download: (p) => (
    <svg viewBox="0 0 24 24" width="18" height="18" {...S} {...p}>
      <path d="M12 4v11M7 10l5 5 5-5M5 20h14" />
    </svg>
  ),
  copy: (p) => (
    <svg viewBox="0 0 24 24" width="18" height="18" {...S} {...p}>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15V5a2 2 0 0 1 2-2h8" />
    </svg>
  ),
  check: (p) => (
    <svg viewBox="0 0 24 24" width="16" height="16" {...S} {...p}>
      <path d="m5 12 5 5L20 7" />
    </svg>
  ),
  warn: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <path d="M12 4 2.8 19.5h18.4L12 4Z" />
      <path d="M12 10v4.5M12 17.3v.2" />
    </svg>
  ),
  image: (p) => (
    <svg viewBox="0 0 24 24" width="22" height="22" {...S} {...p}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="8.5" cy="9.5" r="1.5" />
      <path d="m4 18 5-5 4 4 3-3 4 4" />
    </svg>
  ),
}
