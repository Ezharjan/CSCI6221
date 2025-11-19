import { useEffect } from "react";
import { X } from "lucide-react";

interface ImageModalProps {
  isOpen: boolean;
  src: string;
  caption: string;
  onClose: () => void;
}

export const ImageModal = ({ isOpen, src, caption, onClose }: ImageModalProps) => {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleEscape);
      document.body.style.overflow = "unset";
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm animate-modal-fade-in"
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-white/90 hover:text-white transition-colors duration-300 p-2 hover:bg-white/10 rounded-full z-50"
        aria-label="Close modal"
      >
        <X className="w-8 h-8" />
      </button>
      
      <div 
        className="relative max-w-7xl max-h-[90vh] animate-modal-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={src}
          alt={caption}
          className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl"
        />
        <div className="mt-4 text-center">
          <p className="text-white/90 text-lg font-medium px-4 animate-fade-in" style={{ animationDelay: '200ms' }}>
            {caption}
          </p>
        </div>
      </div>
    </div>
  );
};
