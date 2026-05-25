import { createContext, useContext, useState, type ReactNode } from 'react';
import { DEFAULT_MODEL_ID, getModelById, selectableModels, type ModelOption } from '../lib/models';
import { devToolsEnabled } from '../lib/devTools';

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

  // Req 7: the pretrained baseline is only selectable with dev tools on.
  const devTools = devToolsEnabled();
  const models = selectableModels(devTools);
  const selected = getModelById(selectedId, devTools);
  return (
    <ModelContext.Provider value={{ models, selected, setSelectedId }}>
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
