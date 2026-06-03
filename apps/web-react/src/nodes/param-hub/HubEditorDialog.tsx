import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import type { LGraphNode } from '../../core';
import type { ParamHubMeta, SlotMeta } from './param-protocol';
import './HubEditorDialog.css';

interface HubEditorDialogProps {
  node: LGraphNode;
  meta: ParamHubMeta;
  open: boolean;
  onClose: () => void;
  onRename: (slotName: string, newLabel: string) => void;
  onRemove: (slotName: string) => void;
}

export function HubEditorDialog({
  node,
  meta,
  open,
  onClose,
  onRename,
  onRemove,
}: HubEditorDialogProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const slots = meta.slots;

  // Close on Escape / click outside
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === overlayRef.current) onClose();
  };

  if (!open) return null;

  // Duplicate labels detection
  const labels = new Map<string, number>();
  for (const s of slots) labels.set(s.label, (labels.get(s.label) || 0) + 1);
  const dup = new Set(
    Array.from(labels.entries())
      .filter(([k, c]) => c > 1 && k !== '')
      .map(([k]) => k),
  );

  const isLinked = (slotName: string): boolean => {
    const inputs = node.inputs ?? [];
    const inp = inputs.find((s) => s.name === slotName);
    return !!inp && inp.link != null;
  };

  return createPortal(
    <div
      ref={overlayRef}
      className="sr-hub-dialog-overlay"
      onClick={handleOverlayClick}
    >
      <div className="sr-hub-dialog">
        <div className="sr-hub-dialog-header">
          <span>Edit Sr Param Hub — {node.title || 'ParamHub'}</span>
          <button
            type="button"
            className="sr-hub-dialog-close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="sr-hub-dialog-body">
          {slots.length === 0 && (
            <div className="sr-hub-dialog-empty">
              No input slots. Drag any output onto the empty slot to create one.
            </div>
          )}
          <ul className="sr-hub-dialog-list">
            {slots.map((s) => (
              <SlotRow
                key={s.name}
                slot={s}
                linked={isLinked(s.name)}
                duplicate={dup.has(s.label)}
                onRename={(label) => onRename(s.name, label)}
                onRemove={() => onRemove(s.name)}
              />
            ))}
          </ul>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function SlotRow({
  slot,
  linked,
  duplicate,
  onRename,
  onRemove,
}: {
  slot: SlotMeta;
  linked: boolean;
  duplicate: boolean;
  onRename: (newLabel: string) => void;
  onRemove: () => void;
}) {
  const status: { label: string; cls: 'ok' | 'warn' | 'idle' } = linked
    ? { label: slot.type, cls: 'ok' }
    : { label: 'empty', cls: 'idle' };

  return (
    <li className="sr-hub-dialog-row">
      <input
        className={`sr-hub-dialog-input ${duplicate ? 'sr-hub-dialog-input-error' : ''}`}
        value={slot.label}
        placeholder={slot.name}
        spellCheck={false}
        onChange={(e) => onRename(e.target.value)}
        title={
          duplicate
            ? 'Duplicate label - downstream Sr Param Port will not be able to disambiguate'
            : 'Slot label (used by downstream Sr Param Port to pick this value)'
        }
      />
      <span
        className={`sr-hub-dialog-status ${status.cls}`}
        title={`Slot type: ${slot.type}`}
      >
        {status.label}
      </span>
      <button
        type="button"
        className="sr-hub-dialog-btn-danger"
        title="Delete this slot (will disconnect any link)"
        onClick={onRemove}
      >
        ✕
      </button>
    </li>
  );
}
