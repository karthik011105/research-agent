import { useState } from 'react'
import { KeyRound } from 'lucide-react'
import { api, getApiKey, setApiKey } from '../lib/api'
import { Button } from './ui'

export function ApiKeyDialog({ onClose, required }: { onClose: () => void; required?: boolean }) {
  const [key, setKey] = useState(getApiKey())
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(false)

  async function save() {
    if (!key.trim()) return setError('Enter the app password.')
    setChecking(true); setError('')
    const previous = getApiKey()
    setApiKey(key)
    try {
      await api.get('/stats')
      onClose()
    } catch (e) {
      setApiKey(previous)
      setError(e instanceof Error ? e.message : 'Could not check the key')
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl border border-line bg-panel p-6 shadow-2xl">
        <div className="mb-4 flex items-center gap-3">
          <div className="rounded-lg bg-accent-soft p-2 text-accent"><KeyRound size={20} /></div>
          <div>
            <div className="font-semibold">App password</div>
            <div className="text-sm text-muted">The API_KEY from the research-agent/config secret. It is stored only in this browser.</div>
          </div>
        </div>
        <input type="password" autoFocus value={key} onChange={e => setKey(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && save()} placeholder="Paste the app password"
          className="w-full rounded-lg border border-line bg-bg px-3 py-2.5 text-sm outline-none focus:border-accent" />
        {error && <div className="mt-2 text-sm text-bad">{error}</div>}
        <div className="mt-5 flex justify-end gap-2">
          {!required && <Button variant="ghost" onClick={onClose}>Cancel</Button>}
          <Button variant="primary" onClick={save} disabled={checking}>{checking ? 'Checking...' : 'Save'}</Button>
        </div>
      </div>
    </div>
  )
}
