#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/post_prepml/outputs/inference-runner-%j.out

COMPUTE_QUAVER=TRUE

# hard to do 2 months in one 2 days of sbatch
EXPVER="j1li"
NMEM=10
FIRST_REF_DATE=20230801
LAST_REF_DATE=20230810
DATE_STEP=24
FIRST_LEAD_TIME=24
LAST_LEAD_TIME=240
LEAD_TIME_STEP=24
GRID="O320"

# Load quaver module
module load quaver

# Build parameter string once
PARAMS="--expver ${EXPVER} \
--nmem ${NMEM} \
--first_reference_date ${FIRST_REF_DATE} \
--last_reference_date ${LAST_REF_DATE} \
--date_step ${DATE_STEP} \
--first_lead_time ${FIRST_LEAD_TIME} \
--last_lead_time ${LAST_LEAD_TIME} \
--lead_time_step ${LEAD_TIME_STEP} \
--grid ${GRID}"


# Run quaver computation
if [ "$COMPUTE_QUAVER" = TRUE ]; then
    echo "Quavering q_compute_probabilistic.py..."
    echo "quaver /home/ecm5702/dev/post_prepml/quaver_scoring/quaver_ml/q_compute_probabilistic.py ${PARAMS}"
    quaver /home/ecm5702/dev/post_prepml/quaver_scoring/quaver_ml/q_compute_probabilistic.py ${PARAMS}
fi



echo "All done!"
