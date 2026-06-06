import React, { useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import './HubPanel.css';
import type { LGraphNode } from '../../core';
import type { ParamHubMeta, SlotMeta } from './param-protocol';

interface HubPanelProps {
  meta: ParamHubMeta;
  /** 重命名 (改 label) */
  onRename: (slotName: string, newLabel: string) => void;
  /** 删除一个槽 */
  onRemove: (slotName: string) => void;
  /** 节点引用，用于显示连线状态徽标 */
  node: LGraphNode;
}

interface HubDialogProps extends HubPanelProps {
  open: boolean;
  onClose: () => void;
}

/* ------------------------------------------------------------------ */
/*  紧凑视图（节点 body 内常驻显示，不展开编辑列表）                  */
/* ------------------------------------------------------------------ */
export function HubPanel({ meta }: Pick<HubPanelProps, 'meta'>) {
  return (
    <div className='sr-hub-compact'>
      <span className='sr-hub-hint'>
        {meta.slots.length} param{meta.slots.length !== 1 ? 's' : ''}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  带触发按钮 + 弹窗的封装（用于节点 body 常驻挂载）                  */
/* ------------------------------------------------------------------ */
export function HubDialogWithTrigger({ meta, onRename, onRemove, node }: HubPanelProps) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ nodeId: number | string }>).detail;
      if (detail.nodeId === node.id) {
        setOpen(true);
      }
    };
    window.addEventListener('sr-hub-open-dialog', handler);
    return () => window.removeEventListener('sr-hub-open-dialog', handler);
  }, [node.id]);

  const openDialog = useCallback(() => setOpen(true), []);
  const closeDialog = useCallback(() => setOpen(false), []);

  return (
    <>
      <div
        className='sr-hub-compact'
        style={{ cursor: 'pointer' }}
        onDoubleClick={openDialog}
        title='Double-click or right-click → "Edit params" to open editor'
      >
        <span className='sr-hub-hint'>
          {meta.slots.length} param{meta.slots.length !== 1 ? 's' : ''}
        </span>
        <button
          type='button'
          className='sr-hub-edit-btn'
          onClick={(e) => { e.stopPropagation(); openDialog(); }}
          title='Open editor dialog (right-click node also works)'
        >
          Edit…
        </button>
      </div>

      <HubDialog open={open} onClose={closeDialog} meta={meta} onRename={onRename} onRemove={onRemove} node={node} />
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  纯弹窗（可外部控制 open/close，也可独立使用）                      */
/* ------------------------------------------------------------------ */
export function HubDialog({ meta, onRename, onRemove, node, open, onClose }: HubDialogProps) {
  // ESC 关闭
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  };

  // 点击遮罩层关闭
  const onBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  if (!open) return null;

  const slots = meta.slots;

  const isLinked = (slotName: string): boolean => {
    const inputs = node.inputs ?? [];
    const inp = inputs.find(s => s.name === slotName);
    return !!inp && inp.link != null;
  };

  // 名称重复检测（用 label）
  const labels = new Map<string, number>();
  for (const s of slots) labels.set(s.label, (labels.get(s.label) || 0) + 1);
  const dup = new Set(
    Array.from(labels.entries())
      .filter(([k, c]) => c > 1 && k !== '')
      .map(([k]) => k),
  );

  return createPortal(
    <div className='sr-hub-dialog-backdrop' onMouseDown={onBackdropClick} onKeyDown={handleKeyDown} tabIndex={-1}>
      <div className='sr-hub-dialog' role='dialog' aria-modal='true'>
        <div className='sr-hub-dialog-header'>
          <span className='sr-hub-dialog-title'>Edit Sr Param Hub</span>
          <button
            type='button'
            className='sr-hub-dialog-close'
            onClick={onClose}
            title='Close (Esc)'
          >
            ✕
          </button>
        </div>

        <div className='sr-hub-dialog-body'>
          {slots.length === 0 && (
            <div className='sr-hub-empty'>
              No input slots. Drag any output onto the empty slot to create one.
            </div>
          )}

          <ul className='sr-hub-list'>
            {slots.map(s => (
              <SlotRow
                key={s.name}
                slot={s}
                linked={isLinked(s.name)}
                duplicate={dup.has(s.label)}
                onRename={label => onRename(s.name, label)}
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
    <li className='sr-hub-row'>
      <div className='sr-hub-row-top'>
        <input
          className={`sr-input sr-name ${duplicate ? 'sr-input-error' : ''}`}
          value={slot.label}
          placeholder={slot.name}
          spellCheck={false}
          onChange={e => onRename(e.target.value)}
          title={
            duplicate
              ? 'Duplicate label - downstream Sr Param Port will not be able to disambiguate'
              : 'Slot label (used by downstream Sr Param Port to pick this value)'
          }
        />
        <span className={`sr-port-status ${status.cls}`} title={`Slot type: ${slot.type}`}>
          {status.label}
        </span>
        <div className='sr-hub-actions'>
          <button
            type='button'
            className='sr-btn-icon sr-btn-danger'
            title='Delete this slot (will disconnect any link)'
            onClick={onRemove}
          >
            ✕
          </button>
        </div>
      </div>
    </li>
  );
}
