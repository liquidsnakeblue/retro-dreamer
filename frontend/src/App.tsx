import { useState } from 'react'
import { useTrainingPolling } from './hooks/useTrainingSocket'
import { StatusBar } from './components/StatusBar'
import { EpisodePlayer } from './components/EpisodePlayer'
import { TrainingControls } from './components/TrainingControls'
import { LogTerminal } from './components/LogTerminal'
import { GameSelector } from './components/GameSelector'
import { GameConfigEditor } from './components/GameConfigEditor'

type ActiveTab = 'metrics' | 'config'

export default function App() {
  const [selectedGame, setSelectedGame] = useState('FZero-Snes')
  const [activeTab, setActiveTab] = useState<ActiveTab>('metrics')
  const { connected, status, videos } = useTrainingPolling()

  return (
    <div className="min-h-screen flex flex-col bg-retro-bg">
      {/* Header */}
      <header className="bg-retro-surface border-b border-retro-border px-6 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold tracking-tight">
            <span className="text-retro-accent-bright">RETRO</span>
            <span className="text-retro-text-dim mx-2">/</span>
            <span className="text-retro-text">DreamerV3</span>
          </h1>
          {selectedGame && (
            <>
              <div className="w-px h-4 bg-retro-border" />
              <span className="text-xs text-retro-text-dim font-mono">{selectedGame}</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-retro-success' : 'bg-retro-danger'} animate-pulse`} />
          <span className="text-xs text-retro-text-dim">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      {/* Status Bar */}
      <StatusBar status={status} />

      {/* Main Content */}
      <main className="flex-1 p-4 grid grid-cols-12 gap-4 overflow-hidden" style={{ height: 'calc(100vh - 100px)' }}>
        {/* Left column: Game selector + Controls + Episode Player */}
        <div className="col-span-3 flex flex-col gap-4 overflow-y-auto">
          <GameSelector selectedGame={selectedGame} onSelect={setSelectedGame} />
          <TrainingControls status={status} selectedGame={selectedGame} />
          <EpisodePlayer videos={videos} />
        </div>

        {/* Right column: Tab bar + content */}
        <div className="col-span-9 flex flex-col gap-0 overflow-hidden">
          {/* Tab bar */}
          <div className="bg-retro-surface border border-retro-border rounded-t-lg flex shrink-0">
            <TabButton
              label="Metrics"
              active={activeTab === 'metrics'}
              onClick={() => setActiveTab('metrics')}
            />
            <TabButton
              label="Game Config"
              active={activeTab === 'config'}
              onClick={() => setActiveTab('config')}
            />
          </div>

          {/* Tab content */}
          <div className="flex-1 min-h-0 border-x border-b border-retro-border rounded-b-lg overflow-hidden">
            {activeTab === 'metrics' && (
              <div className="flex flex-col h-full">
                {/* TensorBoard iframe */}
                <div className="bg-retro-card overflow-hidden flex-1 min-h-0">
                  <div className="px-4 py-3 border-b border-retro-border flex items-center justify-between shrink-0">
                    <h2 className="text-sm font-semibold text-retro-text">Training Metrics</h2>
                    <a
                      href="http://localhost:6006"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[10px] text-retro-accent hover:text-retro-accent-bright"
                    >
                      Open full TensorBoard
                    </a>
                  </div>
                  <iframe
                    src="http://localhost:6006"
                    className="w-full border-0"
                    style={{ height: 'calc(100% - 44px)' }}
                    title="TensorBoard"
                  />
                </div>
                {/* Live SheepRL output */}
                <div className="shrink-0">
                  <LogTerminal />
                </div>
              </div>
            )}

            {activeTab === 'config' && (
              <div className="h-full">
                <GameConfigEditor gameId={selectedGame} />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-5 py-2.5 text-xs font-semibold transition-colors border-b-2 first:rounded-tl-lg ${
        active
          ? 'text-retro-accent-bright border-retro-accent bg-retro-card'
          : 'text-retro-text-dim border-transparent hover:text-retro-text hover:bg-retro-card/50'
      }`}
    >
      {label}
    </button>
  )
}
