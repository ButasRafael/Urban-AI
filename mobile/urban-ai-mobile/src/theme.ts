import { createTheme } from '@shopify/restyle'

export const palette = {
  light: {
    transparent: 'transparent',
    primary100: '#E8F6F3',
    primary300: '#A8E0D6',
    primary500: '#16A085',
    primary700: '#0E7C64',
    primary900: '#064D3D',

    secondary100: '#F2F7FA',
    secondary300: '#C7DCEC',
    secondary500: '#2C3E50',
    secondary700: '#1F2C38',
    secondary900: '#0F161C',

    success: '#2ecc71',
    warning: '#f39c12',
    error:   '#c0392b',
    info:    '#3498db',

    surface0:   '#FFFFFF',
    surface100: '#ECF0F1',
    onSurface:  '#34495E',

    overlay: 'rgba(0,0,0,0.45)'
  },

  dark: {
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

    overlay: 'rgba(0,0,0,0.45)',
  },
} as const

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

export const textVariants = {
  defaults: { fontFamily: 'Inter_400Regular', fontSize: 16, lineHeight: 24 },
  display: { fontFamily: 'Inter_700Bold',   fontSize: 32, lineHeight: 40 },
  title:   { fontFamily: 'Inter_600SemiBold',fontSize: 22, lineHeight: 28 },
  body:    { fontFamily: 'Inter_400Regular', fontSize: 16, lineHeight: 24 },
  label:   { fontFamily: 'Inter_500Medium',  fontSize: 14, lineHeight: 20 },
} as const

export const lightTheme = createTheme({
  colors: {
    ...palette.light,
    background: palette.light.surface100,
    card:       palette.light.surface0,
    text:       palette.light.onSurface,
    muted:      palette.light.secondary300,
  },
  spacing,
  borderRadii: radii,
  textVariants,
})

export const darkTheme = createTheme({
  colors: {
    ...palette.dark,
    background: palette.dark.surface0,
    card:       palette.dark.surface100,
    text:       palette.dark.onSurface,
    muted:      palette.dark.secondary500,
  },
  spacing,
  borderRadii: radii,
  textVariants,
})

export type Theme = typeof lightTheme
