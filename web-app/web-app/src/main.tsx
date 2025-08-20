import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './auth/AuthContext';
import App from './App';
import './styles/reset.css';
import './styles/variables.css';
import './styles/app.css';
import './styles/auth.css';

const qc = new QueryClient();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <AuthProvider>
        <App /> 
      </AuthProvider>
    </QueryClientProvider>
  </React.StrictMode>
);

