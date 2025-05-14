// src/screens/TestRefreshScreen.tsx
import React from 'react'
import { View, Button, Text } from 'react-native'
import AsyncStorage from '@react-native-async-storage/async-storage'
import client from '../api/client'

export default function TestRefreshScreen() {
  const [log, setLog] = React.useState<string[]>([])

  const append = (msg: string) =>
    setLog(l => [msg, ...l].slice(0, 20))

  const onTest = async () => {
    append('▶️ Forcing invalid access token')
    await AsyncStorage.setItem('accessToken', 'invalid')
    // assume refreshToken in storage is valid
    append('▶️ Calling /infer/list')
    client.get('/infer/list')
      .then(r => append(`✅ Success, got ${r.data.length} items`))
      .catch(e => append(`❌ Failed: ${e.message}`))
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Button title="Test 401→Refresh" onPress={onTest} />
      {log.map((l,i) => <Text key={i}>{l}</Text>)}
    </View>
  )
}
