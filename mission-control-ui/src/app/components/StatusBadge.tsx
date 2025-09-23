// mission-control-ui/src/app/components/StatusBadge.tsx

import { CheckCircle, XCircle, Loader, PauseCircle, Clock, BrainCircuit, Lightbulb, Combine, MessageCircleQuestion } from 'lucide-react'; // Import MessageCircleQuestion

const statusConfig: { [key: string]: { icon: React.ReactNode; color: string; label?: string; } } = {
  pending: { icon: <Clock size={14} />, color: 'bg-gray-500' },
  running: { icon: <Loader size={14} className="animate-spin" />, color: 'bg-blue-500' },
  complete: { icon: <CheckCircle size={14} />, color: 'bg-green-500' },
  failed: { icon: <XCircle size={14} />, color: 'bg-red-500' },
  paused_by_human: { icon: <PauseCircle size={14} />, color: 'bg-yellow-500', label: 'Paused' },
  planning: { icon: <BrainCircuit size={14} />, color: 'bg-purple-500' },
  proposing: { icon: <Lightbulb size={14} />, color: 'bg-indigo-500' },
  waiting_for_children: { icon: <Combine size={14} />, color: 'bg-teal-500', label: 'Waiting' },
  waiting_for_user_response: { icon: <MessageCircleQuestion size={14} />, color: 'bg-orange-500', label: 'Question' }, // New status
};

export const StatusBadge = ({ status }: { status: string }) => {
  const config = statusConfig[status] || statusConfig.pending;
  const label = config.label || status.replace(/_/g, ' ');
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-1 text-xs font-medium text-white ${config.color}`}>
      {config.icon}
      <span className="capitalize">{label}</span>
    </span>
  );
};