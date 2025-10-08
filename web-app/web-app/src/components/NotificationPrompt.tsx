import { useState, useEffect } from 'react';
import '../styles/notification-prompt.css';

interface NotificationPromptProps {
  onAllow: () => void;
  onDeny: () => void;
  onClose: () => void;
}

export default function NotificationPrompt({ onAllow, onDeny, onClose }: NotificationPromptProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Trigger animation after mount
    requestAnimationFrame(() => setIsVisible(true));
  }, []);

  const handleAllow = () => {
    setIsVisible(false);
    setTimeout(onAllow, 200);
  };

  const handleDeny = () => {
    setIsVisible(false);
    setTimeout(onDeny, 200);
  };

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(onClose, 200);
  };

  return (
    <>
      {/* Backdrop overlay */}
      <div
        className={`notification-prompt-overlay ${isVisible ? 'visible' : ''}`}
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Prompt card */}
      <div
        className={`notification-prompt-card ${isVisible ? 'visible' : ''}`}
        role="dialog"
        aria-labelledby="notification-prompt-title"
        aria-describedby="notification-prompt-description"
      >
        {/* Close button */}
        <button
          className="notification-prompt-close"
          onClick={handleClose}
          aria-label="Close"
          title="Close"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>

        {/* Icon */}
        <div className="notification-prompt-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </svg>
        </div>

        {/* Content */}
        <div className="notification-prompt-content">
          <h3 id="notification-prompt-title" className="notification-prompt-title">
            Enable Notifications
          </h3>
          <p id="notification-prompt-description" className="notification-prompt-description">
            Stay updated with real-time alerts about new issues, status changes, and important updates in your area.
          </p>

          {/* Features list */}
          <ul className="notification-prompt-features">
            <li>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M20 6 9 17l-5-5" />
              </svg>
              <span>Real-time issue alerts</span>
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M20 6 9 17l-5-5" />
              </svg>
              <span>Status updates</span>
            </li>
            <li>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M20 6 9 17l-5-5" />
              </svg>
              <span>Priority notifications</span>
            </li>
          </ul>
        </div>

        {/* Actions */}
        <div className="notification-prompt-actions">
          <button
            className="btn btn-primary btn-block"
            onClick={handleAllow}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M20 6 9 17l-5-5" />
            </svg>
            Allow Notifications
          </button>
          <button
            className="btn btn-ghost btn-block"
            onClick={handleDeny}
          >
            Not Now
          </button>
        </div>

        {/* Privacy note */}
        <p className="notification-prompt-privacy">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 3l8 4v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7l8-4Z" />
          </svg>
          You can change this in settings anytime
        </p>
      </div>
    </>
  );
}