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

export const shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 10,
    elevation: 6,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.16,
    shadowRadius: 24,
    elevation: 12,
  },
} as const

export const gradients = {
  primary: ['#A8E0D6', '#16A085'],
  secondary: ['#C7DCEC', '#2C3E50'],
  hero: ['rgba(232, 246, 243, 0.8)', 'rgba(168, 224, 214, 0.6)'],
  heroDark: ['rgba(6, 77, 61, 0.8)', 'rgba(14, 124, 100, 0.6)'],
} as const

export const textVariants = {
  defaults: { fontFamily: 'Inter_400Regular', fontSize: 16, lineHeight: 24 },
  display: { fontFamily: 'Inter_700Bold', fontSize: 32, lineHeight: 40, letterSpacing: 0.2 },
  hero: { fontFamily: 'Inter_800ExtraBold', fontSize: 28, lineHeight: 36, letterSpacing: 0.2 },
  title: { fontFamily: 'Inter_600SemiBold', fontSize: 22, lineHeight: 28 },
  subtitle: { fontFamily: 'Inter_500Medium', fontSize: 18, lineHeight: 24 },
  body: { fontFamily: 'Inter_400Regular', fontSize: 16, lineHeight: 24 },
  label: { fontFamily: 'Inter_500Medium', fontSize: 14, lineHeight: 20 },
  caption: { fontFamily: 'Inter_400Regular', fontSize: 12, lineHeight: 16 },
} as const

export const lightTheme = createTheme({
  mode: 'light' as const,
  colors: {
    ...palette.light,
    background: palette.light.surface100,
    card: palette.light.surface0,
    text: palette.light.onSurface,
    muted: palette.light.secondary300,
    surface50: '#F8FAFA',
    surface200: '#E5E7EB',
    surface300: '#D1D5DB',

    border: palette.light.secondary100,
    borderFocus: palette.light.primary500,
  },
  spacing,
  borderRadii: radii,
  textVariants,
  shadows,
})

export const darkTheme = createTheme({
  mode: 'dark' as const,
  colors: {
    ...palette.dark,
    background: palette.dark.surface0,
    card: palette.dark.surface100,
    text: palette.dark.onSurface,
    muted: palette.dark.secondary500,

    surface50: '#0B1220',
    surface200: '#1F2C38',
    surface300: '#2A3A4A',

    border: palette.dark.secondary300,
    borderFocus: palette.dark.primary500,
  },
  spacing,
  borderRadii: radii,
  textVariants,
  shadows,
})

export type Theme = typeof lightTheme
