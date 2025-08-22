import tokens from '../tokens/brand-tokens.json';

export const opacity = tokens.opacity ?? {
  overlayWeak: 0.08,
  overlayStrong: 0.15,
};

export const alpha = (hex: string, a = 1) => {
  if (!hex) return `rgba(0,0,0,${a})`;
  const h = hex.replace('#', '');
  const isShort = h.length === 3;
  const full = isShort ? h.split('').map(c => c + c).join('') : h;
  const num = parseInt(full, 16);
  const r = (num >> 16) & 255;
  const g = (num >> 8) & 255;
  const b = num & 255;
  return `rgba(${r},${g},${b},${a})`;
};
