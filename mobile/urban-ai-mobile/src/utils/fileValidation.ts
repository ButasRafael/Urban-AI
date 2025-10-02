/**
 * File validation utilities for upload limits
 */

// File size limits (matching backend)
export const MAX_IMAGE_SIZE_MB = 10;
export const MAX_VIDEO_SIZE_MB = 50;
export const MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024;
export const MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024;

// Video duration limits
export const MAX_VIDEO_DURATION_SECONDS = 60;

// Resolution limits
export const MAX_IMAGE_DIMENSION = 4096; // 4K
export const MAX_VIDEO_DIMENSION = 1920; // 1080p

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

/**
 * Validate file size
 */
export async function validateFileSize(
  uri: string,
  isVideo: boolean
): Promise<ValidationResult> {
  try {
    const response = await fetch(uri);
    const blob = await response.blob();
    const sizeBytes = blob.size;

    const maxSize = isVideo ? MAX_VIDEO_SIZE_BYTES : MAX_IMAGE_SIZE_BYTES;
    const maxSizeMB = isVideo ? MAX_VIDEO_SIZE_MB : MAX_IMAGE_SIZE_MB;

    if (sizeBytes > maxSize) {
      const actualSizeMB = (sizeBytes / (1024 * 1024)).toFixed(1);
      return {
        valid: false,
        error: `Fișierul este prea mare: ${actualSizeMB}MB. Limita este ${maxSizeMB}MB.`
      };
    }

    return { valid: true };
  } catch (error) {
    console.error('Failed to validate file size:', error);
    // Don't block upload if validation fails
    return { valid: true };
  }
}

/**
 * Validate image dimensions
 */
export function validateImageDimensions(
  width: number,
  height: number
): ValidationResult {
  if (width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION) {
    return {
      valid: false,
      error: `Imaginea este prea mare: ${width}x${height}px. Dimensiunea maximă este ${MAX_IMAGE_DIMENSION}px.`
    };
  }
  return { valid: true };
}

/**
 * Validate video properties
 */
export function validateVideoProperties(
  duration: number,
  width?: number,
  height?: number
): ValidationResult {
  if (duration > MAX_VIDEO_DURATION_SECONDS) {
    return {
      valid: false,
      error: `Video prea lung: ${duration.toFixed(0)}s. Limita este ${MAX_VIDEO_DURATION_SECONDS}s.`
    };
  }

  if (width && height) {
    if (width > MAX_VIDEO_DIMENSION || height > MAX_VIDEO_DIMENSION) {
      return {
        valid: false,
        error: `Rezoluție prea mare: ${width}x${height}. Maxim ${MAX_VIDEO_DIMENSION}p permis.`
      };
    }
  }

  return { valid: true };
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Check if error is a rate limit error
 */
export function isRateLimitError(error: any): boolean {
  return error?.response?.status === 429;
}

/**
 * Extract rate limit info from error response
 */
export function getRateLimitInfo(error: any): { retryAfter?: number; limit?: string } {
  if (!isRateLimitError(error)) return {};

  const headers = error?.response?.headers;
  if (!headers) return {};

  return {
    retryAfter: headers['retry-after'] ? parseInt(headers['retry-after']) : undefined,
    limit: headers['x-ratelimit-limit']
  };
}

/**
 * Format rate limit error message
 */
export function formatRateLimitError(error: any): string {
  if (!isRateLimitError(error)) return 'Eroare necunoscută';

  const info = getRateLimitInfo(error);
  if (info.retryAfter) {
    return `Prea multe cereri. Încearcă din nou în ${info.retryAfter} secunde.`;
  }

  return 'Prea multe cereri. Te rugăm să aștepți înainte să încerci din nou.';
}