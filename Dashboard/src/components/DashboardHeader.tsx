interface DashboardHeaderProps {
  generatedDate: string;
}

export const DashboardHeader = ({ generatedDate }: DashboardHeaderProps) => {
  return (
    <header className="mb-8 animate-fade-in">
      <div className="relative overflow-hidden rounded-2xl bg-gradient-primary p-8 shadow-lg">
        <div className="relative z-10">
          <h1 className="text-4xl font-bold text-primary-foreground mb-2">
            Visualization Dashboard
          </h1>
          <p className="text-primary-foreground/90 text-lg">
            Generated on: {new Date(generatedDate).toLocaleString('en-US', {
              year: 'numeric',
              month: 'long',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
              timeZone: 'UTC',
              timeZoneName: 'short'
            })}
          </p>
        </div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl -mr-32 -mt-32"></div>
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/10 rounded-full blur-3xl -ml-24 -mb-24"></div>
      </div>
    </header>
  );
};
