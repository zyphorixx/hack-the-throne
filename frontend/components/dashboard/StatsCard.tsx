
import { Card, CardContent } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";

interface StatsCardProps {
    title: string;
    value: string | number;
    icon: LucideIcon;
    description?: string;
}

export function StatsCard({ title, value, icon: Icon, description }: StatsCardProps) {
    return (
        <Card>
            <CardContent className="flex items-center p-6">
                <div className="flex items-center justify-center w-12 h-12 rounded-full bg-primary/10">
                    <Icon className="w-6 h-6 text-primary" />
                </div>
                <div className="ml-4">
                    <p className="text-sm font-medium text-muted-foreground">{title}</p>
                    <h3 className="text-2xl font-bold">{value}</h3>
                    {description && (
                        <p className="text-xs text-muted-foreground mt-1">{description}</p>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}
