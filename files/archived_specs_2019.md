ARCHIVED DOCUMENT
This document reflects an experimental architecture used in early prototypes.

Do not rely on this specification for current system behavior.

Experimental financial structure

In early versions of the system, invoices could be attached to blocks instead of units.

This allowed easier aggregation of costs per building.

However, the model caused problems when different units within the same block had different buyers.

Example:

Block A
Unit A1 buyer
Unit A2 buyer

If invoices were attached to the block, allocating costs between buyers became difficult.

Because of this issue, the system was redesigned.

Invoices are now attached to units only.
