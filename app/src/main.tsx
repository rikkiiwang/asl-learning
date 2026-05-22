import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { SessionProvider } from './auth/SessionProvider';
import { ModelProvider } from './models/ModelProvider';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <SessionProvider>
        <ModelProvider>
          <App />
        </ModelProvider>
      </SessionProvider>
    </BrowserRouter>
  </StrictMode>,
);
