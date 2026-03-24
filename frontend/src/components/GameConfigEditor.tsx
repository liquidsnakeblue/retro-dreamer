import { useState, useEffect, useCallback } from 'react'

const API = '/api'

type ConfigFilename = 'training.json' | 'scenario.json' | 'actions.json' | 'data.json' | 'metadata.json'

const CONFIG_TABS: ConfigFilename[] = [
  'training.json',
  'scenario.json',
  'actions.json',
  'data.json',
  'metadata.json',
]

interface GameConfigEditorProps {
  gameId: string
}

// ─── Types for actions.json ────────────────────────────────────────────────

/** A single action row: array of button booleans */
type ActionRow = boolean[]

/** Parsed actions.json structure: array of action rows */
type ActionsData = ActionRow[]

// ─── Metadata button layout ────────────────────────────────────────────────

function useButtonLayout(gameId: string) {
  const [buttons, setButtons] = useState<string[]>([])

  useEffect(() => {
    if (!gameId) return
    let cancelled = false
    async function fetch_() {
      try {
        const res = await fetch(`${API}/games/${encodeURIComponent(gameId)}/config/metadata.json`)
        if (res.ok && !cancelled) {
          const data = await res.json()
          // gym-retro metadata.json has a "buttons" field at top level
          const layout: string[] = data?.buttons ?? data?.button_layout ?? []
          setButtons(layout)
        }
      } catch {}
    }
    fetch_()
    return () => { cancelled = true }
  }, [gameId])

  return buttons
}

// ─── Actions editor sub-component ─────────────────────────────────────────

interface ActionsEditorProps {
  gameId: string
  buttons: string[]
}

function ActionsEditor({ gameId, buttons }: ActionsEditorProps) {
  const [actions, setActions] = useState<{ name: string; buttons: ActionRow }[]>([])
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [saveOk, setSaveOk] = useState(false)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    setSaveError('')
    setSaveOk(false)
    try {
      const res = await fetch(`${API}/games/${encodeURIComponent(gameId)}/config/actions.json`)
      if (res.ok) {
        const raw = await res.json()
        // Support both formats: {actions: [{name, buttons}]} and flat array
        const actionsArray = raw.actions || raw
        setActions(
          actionsArray.map((item: any, i: number) => {
            if (item.buttons !== undefined) {
              return { name: item.name || `Action ${i}`, buttons: item.buttons.map(Boolean) }
            }
            // Flat array format
            return { name: `Action ${i}`, buttons: (item as number[]).map(Boolean) }
          })
        )
      } else {
        setSaveError(`Failed to load actions.json: ${res.status}`)
      }
    } catch (e) {
      setSaveError(`Load error: ${String(e)}`)
    } finally {
      setLoading(false)
    }
  }, [gameId])

  useEffect(() => { load() }, [load])

  function toggleButton(actionIdx: number, btnIdx: number) {
    setActions(prev => prev.map((a, i) => {
      if (i !== actionIdx) return a
      const newButtons = [...a.buttons]
      newButtons[btnIdx] = !newButtons[btnIdx]
      return { ...a, buttons: newButtons }
    }))
    setSaveOk(false)
  }

  function updateName(actionIdx: number, name: string) {
    setActions(prev => prev.map((a, i) => i === actionIdx ? { ...a, name } : a))
    setSaveOk(false)
  }

  function addAction() {
    setActions(prev => [
      ...prev,
      { name: `Action ${prev.length}`, buttons: Array(buttons.length).fill(false) },
    ])
    setSaveOk(false)
  }

  function removeAction(idx: number) {
    setActions(prev => prev.filter((_, i) => i !== idx))
    setSaveOk(false)
  }

  async function handleSave() {
    setSaving(true)
    setSaveError('')
    setSaveOk(false)
    try {
      const payload = {
        actions: actions.map(a => ({
          name: a.name,
          buttons: a.buttons.map(b => b ? 1 : 0),
        }))
      }
      const res = await fetch(`${API}/games/${encodeURIComponent(gameId)}/config/actions.json`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (res.ok) {
        setSaveOk(true)
      } else {
        const txt = await res.text()
        setSaveError(`Save failed (${res.status}): ${txt}`)
      }
    } catch (e) {
      setSaveError(`Save error: ${String(e)}`)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-retro-text-dim text-xs">
        Loading actions...
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-retro-border shrink-0">
        <button
          onClick={addAction}
          className="text-xs px-3 py-1.5 bg-retro-accent hover:bg-blue-600 text-white rounded transition-colors"
        >
          + Add Action
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="text-xs px-3 py-1.5 bg-retro-success hover:bg-emerald-600 text-white rounded transition-colors disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button
          onClick={load}
          className="text-xs px-3 py-1.5 bg-retro-surface border border-retro-border hover:border-retro-accent text-retro-text rounded transition-colors"
        >
          Reset
        </button>
        {saveOk && <span className="text-xs text-retro-success">Saved!</span>}
        {saveError && <span className="text-xs text-retro-danger">{saveError}</span>}
      </div>

      <div className="flex-1 overflow-auto min-h-0">
        {actions.length === 0 ? (
          <div className="flex items-center justify-center h-full text-retro-text-dim text-xs">
            No actions defined. Click "+ Add Action" to create one.
          </div>
        ) : (
          <table className="w-full text-xs border-collapse">
            <thead className="sticky top-0 z-10 bg-retro-surface">
              <tr>
                <th className="text-left px-3 py-2 text-retro-text-dim font-semibold border-b border-retro-border w-8">#</th>
                <th className="text-left px-3 py-2 text-retro-text-dim font-semibold border-b border-retro-border min-w-[120px]">Name</th>
                {buttons.map(btn => (
                  <th
                    key={btn}
                    className="px-2 py-2 text-retro-text-dim font-semibold border-b border-retro-border text-center min-w-[40px] font-mono"
                  >
                    {btn}
                  </th>
                ))}
                <th className="px-2 py-2 border-b border-retro-border w-10" />
              </tr>
            </thead>
            <tbody>
              {actions.map((action, actionIdx) => (
                <tr
                  key={actionIdx}
                  className={actionIdx % 2 === 0 ? 'bg-retro-card' : 'bg-retro-surface/40'}
                >
                  <td className="px-3 py-1.5 text-retro-text-dim tabular-nums">{actionIdx}</td>
                  <td className="px-3 py-1.5">
                    <input
                      type="text"
                      value={action.name}
                      onChange={e => updateName(actionIdx, e.target.value)}
                      className="w-full bg-transparent border-b border-retro-border/50 text-retro-text text-xs py-0.5 focus:outline-none focus:border-retro-accent font-mono"
                    />
                  </td>
                  {buttons.map((btn, btnIdx) => {
                    const checked = action.buttons[btnIdx] ?? false
                    return (
                      <td key={btn} className="px-2 py-1.5 text-center">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleButton(actionIdx, btnIdx)}
                          className="accent-retro-accent w-3.5 h-3.5 cursor-pointer"
                        />
                      </td>
                    )
                  })}
                  <td className="px-2 py-1.5 text-center">
                    <button
                      onClick={() => removeAction(actionIdx)}
                      className="text-retro-text-dim hover:text-retro-danger transition-colors text-[11px]"
                      title="Remove action"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

// ─── Generic JSON editor ────────────────────────────────────────────────────

interface JsonEditorProps {
  gameId: string
  filename: ConfigFilename
}

function JsonEditor({ gameId, filename }: JsonEditorProps) {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [saveOk, setSaveOk] = useState(false)
  const [jsonError, setJsonError] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setSaveError('')
    setSaveOk(false)
    setJsonError('')
    try {
      const res = await fetch(`${API}/games/${encodeURIComponent(gameId)}/config/${encodeURIComponent(filename)}`)
      if (res.ok) {
        const data = await res.json()
        setText(JSON.stringify(data, null, 2))
      } else {
        setText('')
        setSaveError(`Failed to load ${filename}: ${res.status}`)
      }
    } catch (e) {
      setText('')
      setSaveError(`Load error: ${String(e)}`)
    } finally {
      setLoading(false)
    }
  }, [gameId, filename])

  useEffect(() => { load() }, [load])

  function handleChange(val: string) {
    setText(val)
    setSaveOk(false)
    try {
      JSON.parse(val)
      setJsonError('')
    } catch (e) {
      setJsonError(String(e))
    }
  }

  async function handleSave() {
    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch (e) {
      setSaveError(`Invalid JSON: ${String(e)}`)
      return
    }

    setSaving(true)
    setSaveError('')
    setSaveOk(false)
    try {
      const res = await fetch(`${API}/games/${encodeURIComponent(gameId)}/config/${encodeURIComponent(filename)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      })
      if (res.ok) {
        setSaveOk(true)
      } else {
        const txt = await res.text()
        setSaveError(`Save failed (${res.status}): ${txt}`)
      }
    } catch (e) {
      setSaveError(`Save error: ${String(e)}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-retro-border shrink-0">
        <button
          onClick={handleSave}
          disabled={saving || !!jsonError || loading}
          className="text-xs px-3 py-1.5 bg-retro-success hover:bg-emerald-600 text-white rounded transition-colors disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button
          onClick={load}
          className="text-xs px-3 py-1.5 bg-retro-surface border border-retro-border hover:border-retro-accent text-retro-text rounded transition-colors"
        >
          Reset
        </button>
        {saveOk && <span className="text-xs text-retro-success">Saved!</span>}
        {(saveError || jsonError) && (
          <span className="text-xs text-retro-danger truncate max-w-xs">
            {saveError || jsonError}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          {jsonError && (
            <span className="text-[10px] text-retro-warning">Invalid JSON</span>
          )}
          {!jsonError && text && (
            <span className="text-[10px] text-retro-success">Valid JSON</span>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 p-3">
        {loading ? (
          <div className="flex items-center justify-center h-full text-retro-text-dim text-xs">
            Loading {filename}...
          </div>
        ) : (
          <textarea
            value={text}
            onChange={e => handleChange(e.target.value)}
            spellCheck={false}
            className={`w-full h-full bg-black/30 text-retro-text font-mono text-xs p-3 rounded border resize-none focus:outline-none focus:border-retro-accent leading-relaxed ${
              jsonError ? 'border-retro-danger/50' : 'border-retro-border'
            }`}
          />
        )}
      </div>
    </div>
  )
}

// ─── Main GameConfigEditor ──────────────────────────────────────────────────

export function GameConfigEditor({ gameId }: GameConfigEditorProps) {
  const [activeTab, setActiveTab] = useState<ConfigFilename>('training.json')
  const buttons = useButtonLayout(gameId)

  return (
    <div className="flex flex-col h-full bg-retro-card">
      <div className="flex border-b border-retro-border shrink-0 overflow-x-auto">
        {CONFIG_TABS.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2.5 text-[11px] font-mono whitespace-nowrap transition-colors border-b-2 ${
              activeTab === tab
                ? 'text-retro-accent-bright border-retro-accent bg-retro-surface/30'
                : 'text-retro-text-dim border-transparent hover:text-retro-text hover:bg-retro-surface/20'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0">
        {activeTab === 'actions.json' ? (
          <ActionsEditor key={gameId} gameId={gameId} buttons={buttons} />
        ) : (
          <JsonEditor key={`${gameId}-${activeTab}`} gameId={gameId} filename={activeTab} />
        )}
      </div>
    </div>
  )
}
