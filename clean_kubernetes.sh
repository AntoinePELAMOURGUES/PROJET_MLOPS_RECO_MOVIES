#!/bin/bash

# Script de nettoyage Kubernetes

echo "========================================"
echo " Nettoyage de l'environnement Kubernetes"
echo "========================================"

# Fonction pour afficher le menu
function display_menu() {
    echo "Choisissez une option :"
    echo "1) Supprimer tous les pods dans un namespace"
    echo "2) Supprimer tous les déploiements dans un namespace"
    echo "3) Supprimer tous les services dans un namespace"
    echo "4) Supprimer tous les volumes persistants dans un namespace"
    echo "5) Supprimer un namespace entier"
    echo "6) Quitter"
}

# Boucle principale
while true; do
    display_menu
    read -p "Entrez votre choix [1-6]: " choice

    case $choice in
        1)
            read -p "Entrez le nom du namespace : " namespace
            echo "Suppression de tous les pods dans le namespace '$namespace'..."
            kubectl delete pods --all -n "$namespace"
            ;;
        2)
            read -p "Entrez le nom du namespace : " namespace
            echo "Suppression de tous les déploiements dans le namespace '$namespace'..."
            kubectl delete deployments --all -n "$namespace"
            ;;
        3)
            read -p "Entrez le nom du namespace : " namespace
            echo "Suppression de tous les services dans le namespace '$namespace'..."
            kubectl delete services --all -n "$namespace"
            ;;
        4)
            read -p "Entrez le nom du namespace : " namespace
            echo "Suppression de tous les volumes persistants dans le namespace '$namespace'..."
            kubectl delete pvc --all -n "$namespace"
            ;;
        5)
            read -p "Entrez le nom du namespace à supprimer : " namespace
            echo "Suppression du namespace '$namespace'..."
            kubectl delete namespace "$namespace"
            ;;
        6)
            echo "Sortie du script."
            exit 0
            ;;
        *)
            echo "Choix invalide. Veuillez réessayer."
            ;;
    esac

    echo "Nettoyage terminé."
    echo ""
done