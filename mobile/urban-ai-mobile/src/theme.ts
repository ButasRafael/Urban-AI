import { createTheme } from '@shopify/restyle'

export const palette = {
  light: {
    transparent: 'transparent',
    primary100: '#E8F6F3',
    primary200: '#C8EEE5',  // interpolated
    primary300: '#A8E0D6',
    primary400: '#5FC5B0',  // interpolated
    primary500: '#16A085',
    primary600: '#128E75',  // interpolated
    primary700: '#0E7C64',
    primary800: '#0A6551',  // interpolated
    primary900: '#064D3D',

    secondary100: '#F2F7FA',
    secondary200: '#DCEAF5',  // interpolated
    secondary300: '#C7DCEC',
    secondary400: '#7A8D9E',  // interpolated
    secondary500: '#2C3E50',
    secondary600: '#263544',  // interpolated
    secondary700: '#1F2C38',
    secondary800: '#17212A',  // interpolated
    secondary900: '#0F161C',

    success: '#2ecc71',
    warning: '#f39c12',
    error:   '#c0392b',
    info:    '#3498db',

    surface0:   '#FFFFFF',
    surface100: '#ECF0F1',
    surface200: '#e5e7eb',
    surface300: '#d1d5db',
    onSurface:  '#34495E',
    muted:      '#95A5A6',

    overlay: 'rgba(0,0,0,0.45)'
  },

  dark: {
    transparent: 'transparent',
    primary100: '#064D3D',
    primary200: '#0A6551',  // interpolated
    primary300: '#0E7C64',
    primary400: '#128E75',  // interpolated
    primary500: '#16A085',
    primary600: '#5FC5B0',  // interpolated
    primary700: '#A8E0D6',
    primary800: '#C8EEE5',  // interpolated
    primary900: '#E8F6F3',

    secondary100: '#0F161C',
    secondary200: '#17212A',  // interpolated
    secondary300: '#1F2C38',
    secondary400: '#263544',  // interpolated
    secondary500: '#2C3E50',
    secondary600: '#7A8D9E',  // interpolated
    secondary700: '#C7DCEC',
    secondary800: '#DCEAF5',  // interpolated
    secondary900: '#F2F7FA',

    success: '#27ae60',
    warning: '#e67e22',
    error:   '#e74c3c',
    info:    '#2980b9',

    surface0:   '#1E1E1E',
    surface100: '#2A2A2A',
    surface200: '#1f2c38',
    surface300: '#2a3a4a',
    onSurface:  '#ECF0F1',
    muted:      '#64727a',

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
  sm: 4,   // matches web --radius-sm
  m:  8,
  md: 8,   // matches web --radius-md
  l:  12,  // matches web --radius-lg (changed from 16)
  lg: 12,  // matches web --radius-lg
  xl: 16,
  pill: 999, // for fully rounded pills/chips
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
  button: { fontFamily: 'Inter_700Bold', fontSize: 14, lineHeight: 20, letterSpacing: 0.2 },
  chipSmall: { fontFamily: 'Inter_800ExtraBold', fontSize: 11, lineHeight: 14 },
  chip: { fontFamily: 'Inter_700Bold', fontSize: 12, lineHeight: 16 },
} as const

// Component design tokens matching web app patterns
export const components = {
  // Button styles matching web (btn.css)
  button: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 12,
    fontWeight: '700' as const,
    fontSize: 14,
    letterSpacing: 0.2,
    // Small variant
    sm: { paddingVertical: 7, paddingHorizontal: 10, fontSize: 13, borderRadius: 10 },
    // Large variant
    lg: { paddingVertical: 12, paddingHorizontal: 18, fontSize: 15, borderRadius: 14 },
  },
  // Card styles matching web (issue-modal.css, notifications.css)
  card: {
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
  },
  // Input styles matching web (input.css)
  input: {
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
    fontSize: 14,
    // Small variant
    sm: { paddingVertical: 7, paddingHorizontal: 10, borderRadius: 10 },
    // Large variant
    lg: { paddingVertical: 12, paddingHorizontal: 14, borderRadius: 14 },
  },
  // Chip/Pill styles matching web (issue-modal.css)
  chip: {
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 999,
    fontSize: 12,
    fontWeight: '700' as const,
    borderWidth: 1,
  },
  // Badge styles matching web (notifications.css)
  badge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 999,
    fontSize: 11,
    fontWeight: '800' as const,
  },
  // Modal styles matching web (issue-modal.css)
  modal: {
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
  },
  // Toast styles matching web (toast.css)
  toast: {
    borderRadius: 14,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderWidth: 1,
  },
  // Icon button styles
  iconButton: {
    size: 38,
    borderRadius: 10,
    sm: { size: 32, borderRadius: 8 },
    lg: { size: 44, borderRadius: 12 },
  },
} as const

export const lightTheme = createTheme({
  mode: 'light' as const,
  colors: {
    ...palette.light,
    background: palette.light.surface100,
    card: palette.light.surface0,
    text: palette.light.onSurface,

    // Additional surface levels for component hierarchy
    surface50: '#F8FAFA',

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

    // Additional surface levels for component hierarchy
    surface50: '#0B1220',

    border: palette.dark.secondary300,
    borderFocus: palette.dark.primary500,
  },
  spacing,
  borderRadii: radii,
  textVariants,
  shadows,
})

export type Theme = typeof lightTheme
