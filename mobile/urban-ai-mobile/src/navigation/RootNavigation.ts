
import { 
    createNavigationContainerRef, 
    CommonActions 
  } from '@react-navigation/native'
  import type { RootStackParamList } from './types'
  
  export const navigationRef = createNavigationContainerRef<RootStackParamList>()
  
  export function resetToLogin() {
    if (navigationRef.isReady()) {
      navigationRef.dispatch(
        CommonActions.reset({
          index: 0,
          routes: [{ name: 'Login' }],
        })
      )
    }
  }
  