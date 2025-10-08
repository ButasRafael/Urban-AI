export type RootStackParamList = {
  Register: undefined
  Login: undefined
  VerifyEmail: {
    email?: string
    token?: string
  }
  ForgotPassword: undefined
  ResetPassword: {
    token: string
  }
  Home: undefined
  Gallery: undefined
  Processing: {
    taskId?: string  // Optional to handle race conditions
    mediaId: number
  }
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
      summary_description?: string
      summary_solution?: string
    }
    showInfo: boolean
  }
  NotificationSettings: undefined
  Notifications: undefined
}
