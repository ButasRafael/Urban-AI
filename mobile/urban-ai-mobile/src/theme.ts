
import { createTheme } from '@shopify/restyle'

// ----------  Colour palette  ----------
export const palette = {
  light: {
    // Brand – primary (sea‑green)
    transparent: 'transparent',
    primary100: '#E8F6F3',
    primary300: '#A8E0D6',
    primary500: '#16A085',
    primary700: '#0E7C64',
    primary900: '#064D3D',

    // Brand – secondary (slate)
    secondary100: '#F2F7FA',
    secondary300: '#C7DCEC',
    secondary500: '#2C3E50',
    secondary700: '#1F2C38',
    secondary900: '#0F161C',

    // Semantic
    success: '#2ecc71',
    warning: '#f39c12',
    error:   '#c0392b',
    info:    '#3498db',

    // Surfaces / text
    surface0:   '#FFFFFF',   // cards
    surface100: '#ECF0F1',   // app background
    onSurface:  '#34495E',   // default text colour
  },

  dark: {
    // Tone ramp inverted for dark►light ordering
    transparent: 'transparent',
    primary100: '#064D3D',
    primary300: '#0E7C64',
    primary500: '#16A085',
    primary700: '#A8E0D6',
    primary900: '#E8F6F3',

    secondary100: '#0F161C',
    secondary300: '#1F2C38',
    secondary500: '#2C3E50',
    secondary700: '#C7DCEC',
    secondary900: '#F2F7FA',

    success: '#27ae60',
    warning: '#e67e22',
    error:   '#e74c3c',
    info:    '#2980b9',

    surface0:   '#1E1E1E',
    surface100: '#2A2A2A',
    onSurface:  '#ECF0F1',
  },
} as const

// ----------  Spacing & radii  ----------
export const spacing = {
  xs: 4,
  s:  8,
  m: 16,
  l: 24,
  xl:32,
} as const

export const radii = {
  xs: 2,
  s:  4,
  m:  8,
  l: 16,
} as const

// ----------  Typography variants  ----------
// Inter Variable font‑faces are loaded at runtime via expo‑google‑fonts
export const textVariants = {
  defaults: { fontFamily: 'Inter_400Regular', fontSize: 16, lineHeight: 24 },
  display: { fontFamily: 'Inter_700Bold',   fontSize: 32, lineHeight: 40 },
  title:   { fontFamily: 'Inter_600SemiBold',fontSize: 22, lineHeight: 28 },
  body:    { fontFamily: 'Inter_400Regular', fontSize: 16, lineHeight: 24 },
  label:   { fontFamily: 'Inter_500Medium',  fontSize: 14, lineHeight: 20 },
} as const

// ----------  Restyle themes  ----------
export const theme = createTheme({
  colors: {
    ...palette.light,
    // alias tokens expected by Navigation / components
    background: palette.light.surface100,
    card:       palette.light.surface0,
    text:       palette.light.onSurface,
    muted:      palette.light.secondary300,
  },
  spacing,
  borderRadii: radii,
  textVariants,
})

export const darkTheme = {
  ...theme,
  colors: {
    ...palette.dark,
    background: palette.dark.surface0,
    card:       palette.dark.surface100,
    text:       palette.dark.onSurface,
    muted:      palette.dark.secondary300,
  },
} as const

export type Theme = typeof theme
