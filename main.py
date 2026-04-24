import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# 1. φόρτωση δεδομένω απο το excel
FILE_PATH = 'vathmoi_foititvn.xlsx'

print("Φόρτωση δεδομένων...")
data = pd.read_excel(FILE_PATH, header=[0, 1], decimal=',')


# 2. καθαρισμός δεδομενων του excel από μη σωστές τιμες ΝαΝ, - , -1
data = data.replace(["-", " - ", ""], np.nan)
data = data.replace(-1, np.nan)
data = data.apply(pd.to_numeric, errors='coerce')


# 3. καθορίζω τι ειναι οι στήλες του excel
exam_cols = data.columns[0:2]  # A=Τελική, B=Επαναληπτική
mandatory_cols = data.columns[3:15]  # D-O = Υποχρεωτικές
optional_cols = data.columns[16:26]  # Q-Z = Προαιρετικές


# 4. εισαγωγές από το πληκτρολόγιο
print("\n" + "=" * 50)
print("ΕΠΙΛΟΓΕΣ ΑΝΑΛΥΣΗΣ")
print("=" * 50)

# Επιλογή αριθμού clusters
while True:
    try:
        k = int(input("Πόσα clusters θέλεις; (π.χ. 3, 4, 5): ").strip())
        if k >= 2:
            break
        else:
            print("Παρακαλώ βάλε τουλάχιστον 2 clusters")
    except ValueError:
        print("Λάθος! Παρακαλώ βάλε έναν ακέραιο αριθμό (π.χ. 4)")

# Επιλογή εξέτασης
while True:
    exam_choice = input(
        "Θες να συνδυάσεις με Τελική ή Επαναληπτική; (γράψε 'τελικη' ή 'επαναληπτικη'): ").strip().lower()
    if exam_choice in ['τελικη', 'επαναληπτικη']:
        break
    print("Λάθος επιλογή! Γράψε 'τελικη' ή 'επαναληπτικη'")

if exam_choice == 'τελικη':
    exam_col = exam_cols[0]
    exam_name = "Τελική Εξέταση"
else:
    exam_col = exam_cols[1]
    exam_name = "Επαναληπτική Εξέταση"

# Επιλογή για προαιρετικές δραστηριότητες
while True:
    include_optional = input("Να ληφθούν υπόψιν και οι προαιρετικές εργασίες; (ναι/οχι): ").strip().lower()
    if include_optional in ['ναι', 'οχι']:
        break
    print("Λάθος! Γράψε 'ναι' ή 'οχι'")


# 5. ΕΠΙΛΟΓΗ ΣΤΗΛΩΝ ΓΙΑ CLUSTERING
if include_optional == 'ναι':   # Συνδυάζει υποχρεωτικές + προαιρετικές
    features_cols = list(mandatory_cols) + list(optional_cols)
    print(
        f"\nΧρησιμοποιούνται: {len(mandatory_cols)} υποχρεωτικές + {len(optional_cols)} προαιρετικές = {len(features_cols)} συνολικά")
else:
    features_cols = list(mandatory_cols)
    print(f"\nΧρησιμοποιούνται: {len(mandatory_cols)} υποχρεωτικές εργασίες")
mask_exam = data[exam_col].notna() # Δημιουργεί φίλτρο για όσους έχουν βαθμό εξέτασης
data = data.loc[mask_exam].copy() # Κρατά μόνο όσους έχουν βαθμό εξέτασης

# 6. προεργασία για το cliustering
X = data[features_cols].fillna(data[features_cols].mean()) # Αντικαθιστά κενές τιμές με τον μέσο όρο της στήλης

scaler = StandardScaler() # εργαλείο κανονικοποίησης
X_scaled = scaler.fit_transform(X) #κανονικοποιεί τα δεδομένα


# 7. CLUSTERING (ΜΕ ΑΡΙΘΜΟ ΠΟΥ ΕΔΩΣΕ Ο ΧΡΗΣΤΗΣ)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Δημιουργεί μοντέλο K-Means, με 42 ώστε ίδια αποτελεσματα κάθε φορά που τρεχουμε τον κώδικα και με 10 διαφορετικές αρχικοποιήσεις κέντρων
clusters = kmeans.fit_predict(X_scaled)# clustering και βρίσκει τους clusters
data['Cluster'] = clusters # Αποθηκεύει σε ποιο μοτίοβο cluster ανήκει ο κάθε φοιτητής

print(f"\nΟλοκληρώθηκε το clustering με {k} clusters!")


# 8. οπτική παρουσίαση (ΜΕ ΜΕΣΟ ΟΡΟ ΟΛΩΝ ΤΩΝ ΕΡΓΑΣΙΩΝ) των αποτελεσμάτων
mask_exam = data[exam_col].notna() # Δημιουργεί φίλτρο μόνο για όσους έχουν βαθμό εξέτασης

x_plot = data.loc[mask_exam, exam_col] # Επιλέγει τις τιμές της εξέτασης που Θα μπουν στον άξονα Χ του γραφήματος

y_plot = X.loc[mask_exam, features_cols].mean(axis=1) # Υπολογίζει τον μέσο όρο όλων των εργασιών για κάθε φοιτητή (axis=1 σημαίνει "ανά γραμμή" (δηλαδή ανά φοιτητή)) Θα μπουν στον άξονα Υ

clusters_plot = clusters[mask_exam] # Παίρνει τα clusters των φοιτητών που έχουν εξέταση


plt.figure(figsize=(10, 6)) # Δημιουργεί  γράφημα με μέγεθος 10x6

scatter = plt.scatter(
    x_plot,
    y_plot,
    c=clusters_plot,
    cmap='viridis',
    alpha=0.7,
    s=70,
    edgecolors='black',
    linewidth=0.5
)
# x_plot → άξονας Χ (εξέταση)
# y_plot → άξονας Υ (μέσος όρος εργασιών)
# c=clusters_plot → διαφορετικό χρώμα για κάθε cluster ωστε να μπορεί να γίνεται διάκριση μεταξύ διαφορετικών clusters


# τίτλος στο γράφημα
plt.title(
    f"{exam_name} - Μέσος Όρος {len(features_cols)} Εργασιών\n"
    f"({len(x_plot)} φοιτητές, {k} clusters)"
)

# Όνομα άξονα Χ
plt.xlabel(exam_name)

# Όνομα άξονα Υ
plt.ylabel(f"Μέσος Όρος Εργασιών (0-10)")

# Ενεργοποιεί πλέγμα στο γράφημα (alpha=0.3 καθορίζει τη διαφάνεια των γραμμών δηλ πόσο έντονες θα ειναι )
plt.grid(True, alpha=0.3)

# Εμφανίζει legend για τα clusters στη δεξιά στήλη του γραφήματος
plt.colorbar(scatter, label="Cluster")

# Όρια άξονα Χ
plt.xlim(0, 11)

# Όρια άξονα Υ
plt.ylim(0, 11)

# Βελτιώνει τη διάταξη
plt.tight_layout()

# Εμφανίζει το γράφημα
plt.show()




# 9. αποτελεσματα που βγήκαν ανα cluster
print("\n" + "=" * 50) # για τη γραγική  απεικόνιση των αποτελεσμάτων στο run
print(f"ΑΠΟΤΕΛΕΣΜΑΤΑ ΑΝΑ CLUSTER (σύνολο: {k} clusters)")
print("=" * 50)

for i in range(k):
# Επαναλαμβάνει για κάθε cluster

    group_all = data[data['Cluster'] == i] # Επιλέγει όλους τους φοιτητές του cluster


    group_exam = data[(data['Cluster'] == i) & mask_exam]   # Επιλέγει όσους έδωσαν εξέταση


    print(f"\nCluster {i + 1}")

    print(f"  Σύνολο φοιτητών: {len(group_all)}")   # Πλήθος φοιτητών στο cluster


    print(f"  Εξετάστηκαν: {len(group_exam)}") # Πλήθος φοιτητών που έδωσαν εξέταση


    if len(group_all) > 0:

        print(
            f"  Μ.Ο. όλων των εργασιών: "
            f"{group_all[features_cols].mean().mean():.2f}"# Υπολογίζει τον μέσο όρο εργασιών του cluster με δυο δεκαδικά
        )


    if len(group_exam) > 0:

        print(
            f"  Μ.Ο. {exam_name}: "
            f"{group_exam[exam_col].mean():.2f}"    # Υπολογίζει τον μέσο όρο εξέταση με δυο δεκαδικά
        )




# 10. συσχετιση των αποτελεσμάτων για εξαγωγή καλύτερης πληροφορίας
print("\n" + "=" * 50)
print("ΣΥΣΧΕΤΙΣΗ ΕΡΓΑΣΙΩΝ ΜΕ ΕΞΕΤΑΣΗ")
print("=" * 50)

data_exam = data[mask_exam].copy()# Κρατά μόνο όσους έχουν βαθμό εξέτασης


avg_assignments = data_exam[features_cols].mean(axis=1)# Υπολογισμός μέσου όρου εργασιών ανά φοιτητή για κάθε φοιτητή


exam_scores = data_exam[exam_col]    #  Δημιουργεί έναν πίνακα με βαθμούς εξέτασης

# Υπολογισμός Pearson σηματνικό κομμάτι για να δουμε την ποιότητα των αποτελεσμάτων

corr = avg_assignments.corr(exam_scores)
# Υπολογίζει τον συντελεστή Pearson
print(
    f"Συσχέτιση μεταξύ Μ.Ο. εργασιών και {exam_name}: {corr:.3f}" # Εκτυπώνει τη συσχέτιση με 3 δεκαδικά
)



if corr > 0.5:

    print(
        "✓ Θετική συσχέτιση - "
        "Ο μέσος όρος των εργασιών προβλέπει καλά την εξέταση"
    )
elif corr > 0.3:
    print(
        "◯ Μέτρια συσχέτιση - "
        "Υπάρχει σχέση αλλά και άλλοι παράγοντες επηρεάζουν πχ μη σωστη προετοιμασία λόγω ασθένειας"
    )
else:
    print(
        "✗ Ασθενής συσχέτιση - "
        "Οι εργασίες ΔΕΝ προβλέπουν την επίδοση στην εξέταση των φοιτητών"
    )



# παρουσιαση τελικων στοιχείων

# Εμφανίζει όλους τους διαφορετικούς βαθμούς εξέτασης
print("\nΜοναδικές τιμές εξέτασης:")
print(sorted(data[exam_col].unique()))


# Εμφανίζει τα οριστικοποιημένα κέντρα των clusters μετά το πέρας της διεργασίας
print("\nΤελικά κέντρα:")
print(kmeans.cluster_centers_)


print("\nΟλοκληρώθηκε!")
