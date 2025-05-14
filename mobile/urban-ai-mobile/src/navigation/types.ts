export type RootStackParamList = {
  Register: undefined
  Login: undefined
  Home: undefined
  Gallery: undefined
  Detail: {
    media: {
      media_id: number
      media_type: 'image' | 'video'
      annotated_image_url?: string
      annotated_video_url?: string
      created_at?: string
      address: string
      latitude?: number
      longitude?: number
      predicted_classes: string[]
      descriptions?: string[] 
    }
    showInfo: boolean
  }
  TestRefresh: undefined
}
