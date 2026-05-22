import { createContext, useContext, useState, type ReactNode } from 'react';
import { MODELS, DEFAULT_MODEL_ID, getModelById, type ModelOption } from '../lib/models';

const STORAGE_KEY = 'asl.activeModel';

interface ModelContextValue {
  models: ModelOption[];
  selected: ModelOption;
  setSelectedId: (id: string) => void;
}

const ModelContext = createContext<ModelContextValue | null>(null);

export function ModelProvider({ children }: { children: ReactNode }) {
  const [selectedId, setSelectedIdState] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_MODEL_ID;
    } catch {
      return DEFAULT_MODEL_ID;
    }
  });

  const setSelectedId = (id: string) => {
    setSelectedIdState(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* non-persistent fallback */
    }
  };

  const selected = getModelById(selectedId);
  return (
    <ModelContext.Provider value={{ models: MODELS, selected, setSelectedId }}>
      {children}
    </ModelContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useModel() {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error('useModel must be used within a ModelProvider');
  return ctx;
}
