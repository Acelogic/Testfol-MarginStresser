import React, { createContext, useState, useEffect, useContext } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Types
export type Ticker = { symbol: string; percent: string };

export type SavedScenario = {
    id: string;
    name: string;
    tickers: Ticker[];
    portfolioValue: string;
    marginDebt: string;
    interestRate: string;
    maintenanceMargin: string;
};

export type HistoryItem = {
    id: string;
    date: string;
    scenarioName: string; // "Custom Run" or Saved Scenario Name
    resultSummary: {
        finalEquity: number;
        cagr: number;
        maxDrawdown: number;
        leverage: string;
    };
};

export type AppSettings = {
    defaultInterestRate: string;
    defaultMaintenanceMargin: string;
    theme: 'light' | 'dark' | 'system';
};

type AppContextType = {
    scenarios: SavedScenario[];
    history: HistoryItem[];
    settings: AppSettings;
    activeScenarioToLoad: SavedScenario | null; // For Home to pick up
    markScenarioLoaded: () => void;
    loadScenario: (scenario: SavedScenario) => void;
    addScenario: (scenario: Omit<SavedScenario, 'id'>) => void;
    deleteScenario: (id: string) => void;
    addToHistory: (item: Omit<HistoryItem, 'id'>) => void;
    updateSettings: (newSettings: Partial<AppSettings>) => void;
    clearHistory: () => void;
};

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [scenarios, setScenarios] = useState<SavedScenario[]>([]);
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [settings, setSettings] = useState<AppSettings>({
        defaultInterestRate: '8.0',
        defaultMaintenanceMargin: '25.0',
        theme: 'light',
    });
    const [activeScenarioToLoad, setActiveScenarioToLoad] = useState<SavedScenario | null>(null);

    // Load data on mount
    useEffect(() => {
        const loadData = async () => {
            try {
                const storedScenarios = await AsyncStorage.getItem('scenarios');
                const storedHistory = await AsyncStorage.getItem('history');
                const storedSettings = await AsyncStorage.getItem('settings');

                if (storedScenarios) setScenarios(JSON.parse(storedScenarios));
                if (storedHistory) setHistory(JSON.parse(storedHistory));
                if (storedSettings) setSettings(JSON.parse(storedSettings));
            } catch (e) {
                console.error("Failed to load data", e);
            }
        };
        loadData();
    }, []);

    // Save data updates
    const saveScenarios = async (newScenarios: SavedScenario[]) => {
        setScenarios(newScenarios);
        await AsyncStorage.setItem('scenarios', JSON.stringify(newScenarios));
    };

    const saveHistory = async (newHistory: HistoryItem[]) => {
        setHistory(newHistory);
        await AsyncStorage.setItem('history', JSON.stringify(newHistory));
    };

    const saveSettings = async (newSettings: AppSettings) => {
        setSettings(newSettings);
        await AsyncStorage.setItem('settings', JSON.stringify(newSettings));
    };

    // Actions
    const addScenario = (scenario: Omit<SavedScenario, 'id'>) => {
        const newScenario = { ...scenario, id: Date.now().toString() };
        saveScenarios([...scenarios, newScenario]);
    };

    const deleteScenario = (id: string) => {
        saveScenarios(scenarios.filter(s => s.id !== id));
    };

    const addToHistory = (item: Omit<HistoryItem, 'id'>) => {
        const newItem = { ...item, id: Date.now().toString() };
        saveHistory([newItem, ...history]); // Prepend
    };

    const updateSettings = (newSettings: Partial<AppSettings>) => {
        saveSettings({ ...settings, ...newSettings });
    };

    const clearHistory = () => {
        saveHistory([]);
    };

    const loadScenario = (scenario: SavedScenario) => {
        setActiveScenarioToLoad(scenario);
    };

    const markScenarioLoaded = () => {
        setActiveScenarioToLoad(null);
    };

    return (
        <AppContext.Provider value={{
            scenarios, history, settings,
            activeScenarioToLoad, loadScenario, markScenarioLoaded,
            addScenario, deleteScenario, addToHistory, updateSettings, clearHistory
        }}>
            {children}
        </AppContext.Provider>
    );
};

export const useAppContext = () => {
    const context = useContext(AppContext);
    if (!context) throw new Error("useAppContext must be used within AppProvider");
    return context;
};
