// src/App.tsx
import { useEffect, useState } from 'react';
import { Toaster } from 'sonner';
import AppRouter from './router';
import './styles/toast.css';

function getTheme(): 'light' | 'dark' {
  if (typeof document === 'undefined') return 'light';
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

export default function App() {
  const [toastTheme, setToastTheme] = useState<'light' | 'dark'>(getTheme);

  useEffect(() => {
    // Update when the root <html> class changes (e.g., your login toggle)
    const root = document.documentElement;
    const update = () => setToastTheme(getTheme());

    // 1) Listen for a custom "themechange" event (optional but nice)
    window.addEventListener('themechange', update);

    // 2) Also observe class attribute changes (works even without the event)
    const obs = new MutationObserver(update);
    obs.observe(root, { attributes: true, attributeFilter: ['class'] });

    return () => {
      window.removeEventListener('themechange', update);
      obs.disconnect();
    };
  }, []);

  return (
    <>
      <AppRouter />

      <Toaster
        position="top-right"
        closeButton
        richColors
        theme={toastTheme}
        toastOptions={{
          duration: 3200,
          classNames: {
            toast: 'app-toast',
            title: 'app-toast-title',
            description: 'app-toast-desc',
            actionButton: 'btn btn-primary',
            cancelButton: 'btn',
          },
        }}
      />
    </>
  );
}
